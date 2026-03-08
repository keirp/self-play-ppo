/*
 * Pure C implementation of PPO training for Connect 4.
 * Uses Apple Accelerate (BLAS) for matrix operations.
 *
 * Connect 4: 6 rows x 7 columns, gravity-based, 4-in-a-row to win.
 * Observation: 126 dims (6*7*3 channels: my pieces, opp pieces, bias).
 * Actions: 7 (one per column).
 *
 * Network architecture: Modern MLP with residual connections, LayerNorm, and GELU.
 *   Input projection: Linear(126, H)
 *   Residual blocks:  x = x + GELU(Linear(LayerNorm(x)))   [NUM_BLOCKS times]
 *   Output:           LayerNorm(x) -> policy_head, value_head
 */
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ========== Connect 4 Constants ========== */
#define BOARD_ROWS 6
#define BOARD_COLS 7
#define BOARD_SIZE (BOARD_ROWS * BOARD_COLS)  /* 42 */
#define INPUT_DIM (BOARD_SIZE * 3)            /* 126 */
#define POLICY_DIM BOARD_COLS                 /* 7 */
#define VALUE_DIM 1

/* ========== Configurable Network ========== */
#define MAX_LAYERS 8
#define MAX_H 1024

static int H = 512;
static int NUM_BLOCKS = 3;     /* number of residual blocks (= num_layers - 1) */
static int TOTAL_PARAMS = 0;

/* Parameter offsets computed at init */
/* Input projection */
static int OFF_W_IN, OFF_B_IN;
/* Residual blocks: each has LayerNorm(gamma,beta) + Linear(W,b) */
static int LN_G_OFF[MAX_LAYERS], LN_B_OFF[MAX_LAYERS];  /* LayerNorm gamma, beta */
static int BLK_W_OFF[MAX_LAYERS], BLK_B_OFF[MAX_LAYERS]; /* Linear weight, bias */
/* Final LayerNorm before heads */
static int OFF_LN_FINAL_G, OFF_LN_FINAL_B;
/* Output heads */
static int OFF_WP, OFF_BP, OFF_WV, OFF_BV;

/* Max buffer sizes */
#define MAX_BATCH 2048
#define MAX_TRANS 65536

/* Scratch memory */
static float *act_pre_ln[MAX_LAYERS];   /* input to each block's LN (= residual) */
static float *act_post_ln[MAX_LAYERS];  /* output of LN */
static float *act_post_linear[MAX_LAYERS]; /* output of linear (pre-GELU) */
static float *act_post_gelu[MAX_LAYERS];   /* output of GELU (before residual add) */
static float *act_input_proj;           /* output of input projection */
static float *act_final_ln;             /* output of final LayerNorm */
static float *ln_mu[MAX_LAYERS + 1];    /* LN mean per sample */
static float *ln_rstd[MAX_LAYERS + 1];  /* LN 1/std per sample */

static float *d_lp_buf;
static float *d_logits_buf;
static float *d_value_buf;
static float *dh_buf;
static float *dh_buf2;   /* second gradient buffer for residual path */
static float *dz_buf;
static float *grad_buf;
static float *lp_buf;
static float *prob_buf;
static float *logit_buf;
static float *val_buf;
static float *returns_buf;
static float *adv_buf;
static float *adv_norm_buf;
static int   *perm_buf;

/* Epoch-level buffers for full forward pass */
static float *epoch_pre_ln[MAX_LAYERS];
static float *epoch_post_ln[MAX_LAYERS];
static float *epoch_post_linear[MAX_LAYERS];
static float *epoch_post_gelu[MAX_LAYERS];
static float *epoch_input_proj;
static float *epoch_final_ln;
static float *epoch_ln_mu[MAX_LAYERS + 1];
static float *epoch_ln_rstd[MAX_LAYERS + 1];
static float *epoch_logits;
static float *epoch_values;
static float *epoch_lp;
static float *epoch_probs;
static float *ones_vec;

static int inited = 0;

/* Board game scratch */
static float *board_buf;
static int   *col_height_buf;
static float *obs_buf;
static float *vmask_buf;
static float *cplayer_buf;
static int   *done_buf_g;
static float *winner_buf;
static float *agent_player_buf;

/* Temp buffers for interleaved collection */
static float *tmp_obs;
static long long *tmp_actions;
static float *tmp_log_probs;
static float *tmp_values;
static float *tmp_valid_masks;

/* Per-game transition tracking */
static int game_trans_count[MAX_BATCH];
static int game_trans_idx[MAX_BATCH][22];

/* ========== Init ========== */

void c4_init(int hidden_size, int num_layers) {
    if (inited) return;

    H = hidden_size;
    NUM_BLOCKS = num_layers - 1;  /* first "layer" is input projection, rest are residual blocks */
    if (NUM_BLOCKS < 0) NUM_BLOCKS = 0;

    /* Compute parameter offsets */
    int off = 0;

    /* Input projection: Linear(INPUT_DIM, H) */
    OFF_W_IN = off;  off += H * INPUT_DIM;
    OFF_B_IN = off;  off += H;

    /* Residual blocks */
    for (int b = 0; b < NUM_BLOCKS; b++) {
        LN_G_OFF[b] = off;  off += H;     /* LayerNorm gamma */
        LN_B_OFF[b] = off;  off += H;     /* LayerNorm beta */
        BLK_W_OFF[b] = off; off += H * H; /* Linear weight */
        BLK_B_OFF[b] = off; off += H;     /* Linear bias */
    }

    /* Final LayerNorm */
    OFF_LN_FINAL_G = off;  off += H;
    OFF_LN_FINAL_B = off;  off += H;

    /* Output heads */
    OFF_WP = off;  off += POLICY_DIM * H;
    OFF_BP = off;  off += POLICY_DIM;
    OFF_WV = off;  off += VALUE_DIM * H;
    OFF_BV = off;  off += VALUE_DIM;
    TOTAL_PARAMS = off;

    /* Allocate scratch */
    act_input_proj = (float*)calloc(MAX_BATCH * H, sizeof(float));
    act_final_ln = (float*)calloc(MAX_BATCH * H, sizeof(float));
    for (int b = 0; b < NUM_BLOCKS; b++) {
        act_pre_ln[b] = (float*)calloc(MAX_BATCH * H, sizeof(float));
        act_post_ln[b] = (float*)calloc(MAX_BATCH * H, sizeof(float));
        act_post_linear[b] = (float*)calloc(MAX_BATCH * H, sizeof(float));
        act_post_gelu[b] = (float*)calloc(MAX_BATCH * H, sizeof(float));
        ln_mu[b] = (float*)calloc(MAX_BATCH, sizeof(float));
        ln_rstd[b] = (float*)calloc(MAX_BATCH, sizeof(float));
    }
    /* Final LN stats */
    ln_mu[NUM_BLOCKS] = (float*)calloc(MAX_BATCH, sizeof(float));
    ln_rstd[NUM_BLOCKS] = (float*)calloc(MAX_BATCH, sizeof(float));

    d_lp_buf = (float*)calloc(MAX_BATCH * POLICY_DIM, sizeof(float));
    d_logits_buf = (float*)calloc(MAX_BATCH * POLICY_DIM, sizeof(float));
    d_value_buf = (float*)calloc(MAX_BATCH, sizeof(float));
    dh_buf = (float*)calloc(MAX_BATCH * H, sizeof(float));
    dh_buf2 = (float*)calloc(MAX_BATCH * H, sizeof(float));
    dz_buf = (float*)calloc(MAX_BATCH * H, sizeof(float));
    grad_buf = (float*)calloc(TOTAL_PARAMS, sizeof(float));
    lp_buf = (float*)calloc(MAX_BATCH * POLICY_DIM, sizeof(float));
    prob_buf = (float*)calloc(MAX_BATCH * POLICY_DIM, sizeof(float));
    logit_buf = (float*)calloc(MAX_BATCH * POLICY_DIM, sizeof(float));
    val_buf = (float*)calloc(MAX_BATCH, sizeof(float));
    returns_buf = (float*)calloc(MAX_TRANS, sizeof(float));
    adv_buf = (float*)calloc(MAX_TRANS, sizeof(float));
    adv_norm_buf = (float*)calloc(MAX_TRANS, sizeof(float));
    perm_buf = (int*)calloc(MAX_TRANS, sizeof(int));

    board_buf = (float*)calloc(MAX_BATCH * BOARD_SIZE, sizeof(float));
    col_height_buf = (int*)calloc(MAX_BATCH * BOARD_COLS, sizeof(int));
    obs_buf = (float*)calloc(MAX_BATCH * INPUT_DIM, sizeof(float));
    vmask_buf = (float*)calloc(MAX_BATCH * POLICY_DIM, sizeof(float));
    cplayer_buf = (float*)calloc(MAX_BATCH, sizeof(float));
    done_buf_g = (int*)calloc(MAX_BATCH, sizeof(int));
    winner_buf = (float*)calloc(MAX_BATCH, sizeof(float));
    agent_player_buf = (float*)calloc(MAX_BATCH, sizeof(float));

    /* Epoch buffers */
    epoch_input_proj = (float*)calloc(MAX_TRANS * H, sizeof(float));
    epoch_final_ln = (float*)calloc(MAX_TRANS * H, sizeof(float));
    for (int b = 0; b < NUM_BLOCKS; b++) {
        epoch_pre_ln[b] = (float*)calloc(MAX_TRANS * H, sizeof(float));
        epoch_post_ln[b] = (float*)calloc(MAX_TRANS * H, sizeof(float));
        epoch_post_linear[b] = (float*)calloc(MAX_TRANS * H, sizeof(float));
        epoch_post_gelu[b] = (float*)calloc(MAX_TRANS * H, sizeof(float));
        epoch_ln_mu[b] = (float*)calloc(MAX_TRANS, sizeof(float));
        epoch_ln_rstd[b] = (float*)calloc(MAX_TRANS, sizeof(float));
    }
    epoch_ln_mu[NUM_BLOCKS] = (float*)calloc(MAX_TRANS, sizeof(float));
    epoch_ln_rstd[NUM_BLOCKS] = (float*)calloc(MAX_TRANS, sizeof(float));
    epoch_logits = (float*)calloc(MAX_TRANS * POLICY_DIM, sizeof(float));
    epoch_values = (float*)calloc(MAX_TRANS, sizeof(float));
    epoch_lp = (float*)calloc(MAX_TRANS * POLICY_DIM, sizeof(float));
    epoch_probs = (float*)calloc(MAX_TRANS * POLICY_DIM, sizeof(float));
    ones_vec = (float*)calloc(MAX_BATCH, sizeof(float));
    for (int i = 0; i < MAX_BATCH; i++) ones_vec[i] = 1.0f;

    /* Temp collection buffers */
    tmp_obs = (float*)calloc(MAX_TRANS * INPUT_DIM, sizeof(float));
    tmp_actions = (long long*)calloc(MAX_TRANS, sizeof(long long));
    tmp_log_probs = (float*)calloc(MAX_TRANS, sizeof(float));
    tmp_values = (float*)calloc(MAX_TRANS, sizeof(float));
    tmp_valid_masks = (float*)calloc(MAX_TRANS * POLICY_DIM, sizeof(float));

    inited = 1;
}

int c4_total_params(void) { return TOTAL_PARAMS; }

/* ========== RNG (xoshiro128+) ========== */
static uint32_t rng_s[4];

static inline uint32_t rotl(const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static inline uint32_t xoshiro128p(void) {
    const uint32_t result = rng_s[0] + rng_s[3];
    const uint32_t t = rng_s[1] << 9;
    rng_s[2] ^= rng_s[0];
    rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2];
    rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t;
    rng_s[3] = rotl(rng_s[3], 11);
    return result;
}

static inline float rng_float(void) {
    return (float)(xoshiro128p() >> 8) / 16777216.0f;
}

void c4_seed(unsigned long seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        rng_s[i] = (uint32_t)z;
    }
}

/* ========== Core Math ========== */

static inline void linear_fwd(const float *in, const float *W, const float *b,
                               float *out, int B, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, N, K, 1.0f, in, K, W, K, 0.0f, out, N);
    /* Broadcast bias add: out += ones * b^T (rank-1 update) */
    cblas_sger(CblasRowMajor, B, N, 1.0f, ones_vec, 1, b, 1, out, N);
}

/* GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
   Vectorized using vvtanhf from Accelerate vecLib. */
static float gelu_inner_buf[MAX_TRANS * MAX_H];  /* scratch for vectorized GELU */
static float gelu_tanh_buf[MAX_TRANS * MAX_H];

static inline void gelu_ip(float *x, int n) {
    const float c1 = 0.7978845608f;  /* sqrt(2/pi) */
    const float c2 = 0.044715f;
    /* Compute inner = c1 * (x + c2 * x^3) */
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        gelu_inner_buf[i] = c1 * (xi + c2 * xi * xi * xi);
    }
    /* Vectorized tanh */
    vvtanhf(gelu_tanh_buf, gelu_inner_buf, &n);
    /* x = 0.5 * x * (1 + tanh) */
    for (int i = 0; i < n; i++) {
        x[i] = 0.5f * x[i] * (1.0f + gelu_tanh_buf[i]);
    }
}

/* GELU backward: given x (pre-activation) and d_out, compute d_x.
   Vectorized using vvtanhf. */
static inline void gelu_bwd(const float *x, const float *d_out, float *d_x, int n) {
    const float c1 = 0.7978845608f;
    const float c2 = 0.044715f;
    /* Compute inner = c1 * (x + c2 * x^3) and vectorized tanh */
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        gelu_inner_buf[i] = c1 * (xi + c2 * xi * xi * xi);
    }
    vvtanhf(gelu_tanh_buf, gelu_inner_buf, &n);
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        float t = gelu_tanh_buf[i];
        float sech2 = 1.0f - t * t;
        float d_inner = c1 * (1.0f + 3.0f * c2 * xi * xi);
        float d_gelu = 0.5f * (1.0f + t) + 0.5f * xi * sech2 * d_inner;
        d_x[i] = d_out[i] * d_gelu;
    }
}

/* LayerNorm forward: y = gamma * (x - mu) / sqrt(var + eps) + beta
   Stores mu and rstd (1/sqrt(var+eps)) for backward.
   Vectorized with vDSP. */
static float ln_scratch[MAX_BATCH * MAX_H];  /* scratch for LN computation */

static void layernorm_fwd(const float *x, const float *gamma, const float *beta,
                            float *out, float *mu_out, float *rstd_out,
                            int B, int D) {
    const float eps = 1e-5f;
    for (int i = 0; i < B; i++) {
        const float *xi = x + i * D;
        float *yi = out + i * D;
        float *scratch = ln_scratch + i * D;

        /* Mean */
        float mean = 0.0f;
        vDSP_meanv(xi, 1, &mean, D);

        /* Centered: scratch = x - mean */
        float neg_mean = -mean;
        vDSP_vsadd(xi, 1, &neg_mean, scratch, 1, D);

        /* Variance = dot(scratch, scratch) / D */
        float var = 0.0f;
        vDSP_dotpr(scratch, 1, scratch, 1, &var, D);
        var /= D;

        float rstd = 1.0f / sqrtf(var + eps);
        mu_out[i] = mean;
        rstd_out[i] = rstd;

        /* Normalize: scratch *= rstd */
        vDSP_vsmul(scratch, 1, &rstd, scratch, 1, D);

        /* Scale and shift: yi = gamma * scratch + beta */
        vDSP_vma(scratch, 1, gamma, 1, beta, 1, yi, 1, D);
    }
}

/* LayerNorm backward: computes gradients for gamma, beta, and input x.
   dx_out is accumulated (added to existing values). Vectorized with vDSP. */
static float lnb_xhat[MAX_BATCH * MAX_H];   /* normalized x for LN backward */
static float lnb_dy_scaled[MAX_BATCH * MAX_H]; /* d_out * gamma */

static void layernorm_bwd(const float *d_out, const float *x,
                            const float *gamma, const float *mu, const float *rstd,
                            float *dx_out, float *dgamma, float *dbeta,
                            int B, int D) {
    for (int i = 0; i < B; i++) {
        const float *doi = d_out + i * D;
        const float *xi = x + i * D;
        float m = mu[i];
        float rs = rstd[i];
        float *xhat = lnb_xhat + i * D;
        float *dys = lnb_dy_scaled + i * D;

        /* xhat = (x - mu) * rstd */
        float neg_m = -m;
        vDSP_vsadd(xi, 1, &neg_m, xhat, 1, D);
        vDSP_vsmul(xhat, 1, &rs, xhat, 1, D);

        /* dgamma += d_out * xhat, dbeta += d_out */
        vDSP_vma(doi, 1, xhat, 1, dgamma, 1, dgamma, 1, D);  /* dgamma += doi * xhat */
        vDSP_vadd(doi, 1, dbeta, 1, dbeta, 1, D);

        /* dy_scaled = d_out * gamma */
        vDSP_vmul(doi, 1, gamma, 1, dys, 1, D);

        /* sum_dy = sum(dy_scaled), sum_dy_xhat = dot(dy_scaled, xhat) */
        float sum_dy = 0.0f, sum_dy_xhat = 0.0f;
        vDSP_sve(dys, 1, &sum_dy, D);
        vDSP_dotpr(dys, 1, xhat, 1, &sum_dy_xhat, D);

        /* dx += rs * (dy_scaled - inv_D * (sum_dy + xhat * sum_dy_xhat)) */
        float inv_D = 1.0f / D;
        float c1 = -inv_D * sum_dy;
        float c2 = -inv_D * sum_dy_xhat;
        float *dxi = dx_out + i * D;
        /* dxi += rs * (dys + c1 + c2 * xhat) = rs*dys + rs*c1 + rs*c2*xhat */
        float rs_c1 = rs * c1;
        float rs_c2 = rs * c2;
        /* dxi += rs * dys */
        vDSP_vsma(dys, 1, &rs, dxi, 1, dxi, 1, D);
        /* dxi += rs_c1 (scalar add) */
        vDSP_vsadd(dxi, 1, &rs_c1, dxi, 1, D);
        /* dxi += rs_c2 * xhat */
        vDSP_vsma(xhat, 1, &rs_c2, dxi, 1, dxi, 1, D);
    }
}

/* segment_norm_sq removed — grad norm now uses single vDSP_dotpr on full buffer */

/* ========== Forward Passes ========== */

/* Forward for training (stores all activations for backward) */
static void forward_train(const float *params, const float *obs, int B,
                           float *out_logits, float *out_values) {
    /* Input projection */
    linear_fwd(obs, params + OFF_W_IN, params + OFF_B_IN, act_input_proj, B, INPUT_DIM, H);

    /* Residual blocks */
    float *cur = act_input_proj;
    for (int b = 0; b < NUM_BLOCKS; b++) {
        /* Copy input for residual connection */
        memcpy(act_pre_ln[b], cur, B * H * sizeof(float));

        /* LayerNorm */
        layernorm_fwd(cur, params + LN_G_OFF[b], params + LN_B_OFF[b],
                       act_post_ln[b], ln_mu[b], ln_rstd[b], B, H);

        /* Linear */
        linear_fwd(act_post_ln[b], params + BLK_W_OFF[b], params + BLK_B_OFF[b],
                    act_post_linear[b], B, H, H);

        /* GELU */
        memcpy(act_post_gelu[b], act_post_linear[b], B * H * sizeof(float));
        gelu_ip(act_post_gelu[b], B * H);

        /* Residual: act_post_gelu[b] += act_pre_ln[b] */
        int bh = B * H;
        vDSP_vadd(act_post_gelu[b], 1, act_pre_ln[b], 1, act_post_gelu[b], 1, bh);

        cur = act_post_gelu[b];
    }

    /* Final LayerNorm */
    layernorm_fwd(cur, params + OFF_LN_FINAL_G, params + OFF_LN_FINAL_B,
                   act_final_ln, ln_mu[NUM_BLOCKS], ln_rstd[NUM_BLOCKS], B, H);

    /* Output heads */
    linear_fwd(act_final_ln, params + OFF_WP, params + OFF_BP, out_logits, B, H, POLICY_DIM);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, VALUE_DIM, H, 1.0f, act_final_ln, H, params + OFF_WV, H,
                0.0f, out_values, VALUE_DIM);
    float bv = params[OFF_BV];
    for (int i = 0; i < B; i++) out_values[i] += bv;
}

/* Forward for inference (minimal memory, no activation storage) */
static void forward_infer(const float *params, const float *obs, int B,
                            float *out_logits, float *out_values) {
    /* Use act_pre_ln[0] and act_post_ln[0] as ping-pong buffers */
    float *cur = act_pre_ln[0];
    float *tmp = act_post_ln[0];
    float *ln_out = act_post_linear[0];
    float *mu_tmp = ln_mu[0];
    float *rstd_tmp = ln_rstd[0];

    /* Input projection */
    linear_fwd(obs, params + OFF_W_IN, params + OFF_B_IN, cur, B, INPUT_DIM, H);

    /* Residual blocks */
    for (int b = 0; b < NUM_BLOCKS; b++) {
        /* LN */
        layernorm_fwd(cur, params + LN_G_OFF[b], params + LN_B_OFF[b],
                       ln_out, mu_tmp, rstd_tmp, B, H);
        /* Linear */
        linear_fwd(ln_out, params + BLK_W_OFF[b], params + BLK_B_OFF[b],
                    tmp, B, H, H);
        /* GELU */
        gelu_ip(tmp, B * H);
        /* Residual add: cur = cur + tmp */
        vDSP_vadd(cur, 1, tmp, 1, cur, 1, B * H);
    }

    /* Final LN */
    layernorm_fwd(cur, params + OFF_LN_FINAL_G, params + OFF_LN_FINAL_B,
                   ln_out, mu_tmp, rstd_tmp, B, H);

    /* Output heads */
    linear_fwd(ln_out, params + OFF_WP, params + OFF_BP, out_logits, B, H, POLICY_DIM);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, VALUE_DIM, H, 1.0f, ln_out, H, params + OFF_WV, H,
                0.0f, out_values, VALUE_DIM);
    float bv = params[OFF_BV];
    for (int i = 0; i < B; i++) out_values[i] += bv;
}

/* ========== Log-softmax ========== */

static void log_softmax_masked(float *logits, const float *valid_masks,
                                 float *log_probs, float *probs, int B) {
    for (int i = 0; i < B; i++) {
        float *lg = logits + i*POLICY_DIM;
        const float *vm = valid_masks + i*POLICY_DIM;
        float *lp = log_probs + i*POLICY_DIM;
        float *pr = probs + i*POLICY_DIM;

        for (int j = 0; j < POLICY_DIM; j++)
            if (vm[j] < 0.5f) lg[j] = -1e8f;

        float mx = lg[0];
        for (int j = 1; j < POLICY_DIM; j++)
            if (lg[j] > mx) mx = lg[j];

        float s = 0.0f;
        for (int j = 0; j < POLICY_DIM; j++) {
            pr[j] = expf(lg[j] - mx);
            s += pr[j];
        }
        float log_s = logf(s) + mx;
        for (int j = 0; j < POLICY_DIM; j++) {
            pr[j] /= s;
            lp[j] = lg[j] - log_s;
        }
    }
}

/* ========== Backward Pass ========== */

static float backward_batch(const float *params, const float *obs,
                            const long long *actions, const float *old_lp,
                            const float *ret, const float *adv,
                            const float *valid_masks,
                            int B, float clip_eps, float vf_coef, float ent_coef,
                            float *out_pl, float *out_vl, float *out_ent, float *out_kl) {
    forward_train(params, obs, B, logit_buf, val_buf);
    log_softmax_masked(logit_buf, valid_masks, lp_buf, prob_buf, B);

    float policy_loss = 0.0f, value_loss = 0.0f, entropy = 0.0f, approx_kl = 0.0f;
    float inv_B = 1.0f / (float)B;
    memset(d_lp_buf, 0, B * POLICY_DIM * sizeof(float));

    for (int i = 0; i < B; i++) {
        int ai = (int)actions[i];
        float new_lpi = lp_buf[i*POLICY_DIM + ai];
        float ratio = expf(new_lpi - old_lp[i]);

        float surr1 = ratio * adv[i];
        float clipped = ratio < 1-clip_eps ? 1-clip_eps : (ratio > 1+clip_eps ? 1+clip_eps : ratio);
        float surr2 = clipped * adv[i];
        float s = surr1 < surr2 ? surr1 : surr2;
        policy_loss -= s * inv_B;

        float d_new_lp;
        if (surr1 <= surr2)
            d_new_lp = -adv[i] * ratio * inv_B;
        else
            d_new_lp = (clipped == ratio) ? -adv[i] * ratio * inv_B : 0.0f;
        d_lp_buf[i*POLICY_DIM + ai] += d_new_lp;

        float vdiff = val_buf[i] - ret[i];
        value_loss += vdiff * vdiff * inv_B;
        d_value_buf[i] = vf_coef * 2.0f * vdiff * inv_B;

        float ent_i = 0.0f;
        for (int j = 0; j < POLICY_DIM; j++)
            ent_i -= prob_buf[i*POLICY_DIM+j] * lp_buf[i*POLICY_DIM+j];
        entropy += ent_i * inv_B;

        for (int j = 0; j < POLICY_DIM; j++) {
            d_lp_buf[i*POLICY_DIM+j] += ent_coef * inv_B * prob_buf[i*POLICY_DIM+j]
                                          * (lp_buf[i*POLICY_DIM+j] + 1.0f);
        }
        approx_kl += (old_lp[i] - new_lpi) * inv_B;
    }

    *out_pl = policy_loss; *out_vl = value_loss; *out_ent = entropy; *out_kl = approx_kl;

    /* Log-softmax backward */
    for (int i = 0; i < B; i++) {
        float sum_dlp = 0.0f;
        for (int j = 0; j < POLICY_DIM; j++)
            sum_dlp += d_lp_buf[i*POLICY_DIM+j];
        for (int j = 0; j < POLICY_DIM; j++)
            d_logits_buf[i*POLICY_DIM+j] = d_lp_buf[i*POLICY_DIM+j]
                                            - prob_buf[i*POLICY_DIM+j] * sum_dlp;
    }

    /* Zero gradients */
    memset(grad_buf, 0, TOTAL_PARAMS * sizeof(float));

    /* Backward through output heads */
    float *gWp = grad_buf + OFF_WP, *gbp = grad_buf + OFF_BP;
    float *gWv = grad_buf + OFF_WV, *gbv = grad_buf + OFF_BV;

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                POLICY_DIM, H, B, 1.0f, d_logits_buf, POLICY_DIM, act_final_ln, H,
                0.0f, gWp, H);
    cblas_sgemv(CblasRowMajor, CblasTrans, B, POLICY_DIM,
                1.0f, d_logits_buf, POLICY_DIM, ones_vec, 1, 0.0f, gbp, 1);

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                VALUE_DIM, H, B, 1.0f, d_value_buf, VALUE_DIM, act_final_ln, H,
                0.0f, gWv, H);
    { float dv_sum = 0; for (int i = 0; i < B; i++) dv_sum += d_value_buf[i]; gbv[0] = dv_sum; }

    /* dh = d_logits @ Wp + d_value outer Wv */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                B, H, POLICY_DIM, 1.0f, d_logits_buf, POLICY_DIM, params + OFF_WP, H,
                0.0f, dh_buf, H);
    cblas_sger(CblasRowMajor, B, H, 1.0f, d_value_buf, 1, params + OFF_WV, 1, dh_buf, H);

    /* Backward through final LayerNorm */
    /* dh_buf has gradient w.r.t. act_final_ln output */
    /* The input to final LN is the output of the last residual block (or input_proj if no blocks) */
    float *final_ln_input = (NUM_BLOCKS > 0) ? act_post_gelu[NUM_BLOCKS - 1] : act_input_proj;
    memset(dh_buf2, 0, B * H * sizeof(float));
    layernorm_bwd(dh_buf, final_ln_input,
                   params + OFF_LN_FINAL_G, ln_mu[NUM_BLOCKS], ln_rstd[NUM_BLOCKS],
                   dh_buf2, grad_buf + OFF_LN_FINAL_G, grad_buf + OFF_LN_FINAL_B, B, H);
    /* dh_buf2 now has gradient w.r.t. final LN input */
    memcpy(dh_buf, dh_buf2, B * H * sizeof(float));

    /* Backward through residual blocks (reverse order) */
    for (int b = NUM_BLOCKS - 1; b >= 0; b--) {
        /* dh_buf has gradient w.r.t. block output (= pre_ln[b] + post_gelu[b]) */
        /* Residual: d_pre_ln[b] += dh_buf, d_post_gelu[b] = dh_buf */
        /* We'll propagate through the GELU -> Linear -> LN path, then add residual grad */

        /* GELU backward: d_post_linear = gelu_bwd(post_linear[b], dh_buf) */
        gelu_bwd(act_post_linear[b], dh_buf, dz_buf, B * H);

        /* Linear backward: d_post_ln = dz @ W, dW = dz^T @ post_ln, db = sum(dz) */
        float *gW = grad_buf + BLK_W_OFF[b];
        float *gb = grad_buf + BLK_B_OFF[b];
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    H, H, B, 1.0f, dz_buf, H, act_post_ln[b], H,
                    0.0f, gW, H);
        cblas_sgemv(CblasRowMajor, CblasTrans, B, H,
                    1.0f, dz_buf, H, ones_vec, 1, 0.0f, gb, 1);

        /* d_post_ln */
        memset(dh_buf2, 0, B * H * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    B, H, H, 1.0f, dz_buf, H, params + BLK_W_OFF[b], H,
                    0.0f, dh_buf2, H);

        /* LayerNorm backward: accumulates into a fresh buffer */
        float *d_ln_input = dz_buf;  /* reuse dz_buf */
        memset(d_ln_input, 0, B * H * sizeof(float));
        layernorm_bwd(dh_buf2, act_pre_ln[b],
                       params + LN_G_OFF[b], ln_mu[b], ln_rstd[b],
                       d_ln_input, grad_buf + LN_G_OFF[b], grad_buf + LN_B_OFF[b], B, H);

        /* Add residual gradient: dh_buf = d_ln_input + dh_buf (residual path) */
        vDSP_vadd(d_ln_input, 1, dh_buf, 1, dh_buf, 1, B * H);
    }

    /* Backward through input projection */
    /* dh_buf has gradient w.r.t. input projection output */
    float *gW_in = grad_buf + OFF_W_IN;
    float *gb_in = grad_buf + OFF_B_IN;
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                H, INPUT_DIM, B, 1.0f, dh_buf, H, obs, INPUT_DIM,
                0.0f, gW_in, INPUT_DIM);
    cblas_sgemv(CblasRowMajor, CblasTrans, B, H,
                1.0f, dh_buf, H, ones_vec, 1, 0.0f, gb_in, 1);

    /* Compute gradient norm — single dot product over entire grad buffer */
    float gnorm_sq = 0;
    vDSP_dotpr(grad_buf, 1, grad_buf, 1, &gnorm_sq, TOTAL_PARAMS);
    return gnorm_sq;
}

/* ========== Adam ========== */

static void adam_step_prenorm(float *params, float *grads, float *m, float *v,
                               int n, int t, float lr, float max_grad_norm,
                               float gnorm_sq) {
    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    float gnorm = sqrtf(gnorm_sq);
    float clip_scale = (gnorm > max_grad_norm) ? max_grad_norm / (gnorm + 1e-6f) : 1.0f;

    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2_sqrt = sqrtf(1.0f - powf(beta2, (float)t));
    float step_size = lr / bc1;
    float inv_bc2_sqrt = 1.0f / bc2_sqrt;
    float one_minus_beta1 = 1.0f - beta1;
    float one_minus_beta2 = 1.0f - beta2;

    for (int i = 0; i < n; i++) {
        float g = grads[i] * clip_scale;
        float mi = beta1 * m[i] + one_minus_beta1 * g;
        float vi = beta2 * v[i] + one_minus_beta2 * g * g;
        m[i] = mi;
        v[i] = vi;
        params[i] -= step_size * mi / (sqrtf(vi) * inv_bc2_sqrt + eps);
    }
}

/* ========== GAE ========== */

static void compute_gae(const float *rewards, const float *values,
                          const float *dones, int N,
                          float gamma, float lam,
                          float *returns, float *advantages) {
    float last_gae = 0.0f;
    for (int t = N-1; t >= 0; t--) {
        float next_val = (t == N-1 || dones[t] > 0.5f) ? 0.0f : values[t+1];
        float not_done = 1.0f - dones[t];
        float delta = rewards[t] + gamma * next_val * not_done - values[t];
        last_gae = delta + gamma * lam * not_done * last_gae;
        advantages[t] = last_gae;
        returns[t] = last_gae + values[t];
    }
}

/* ========== Shuffle ========== */

static void shuffle(int *arr, int n) {
    for (int i = n-1; i > 0; i--) {
        int j = (int)(rng_float() * (i+1));
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/* ========== PPO Update ========== */

void c4_ppo_update(float *params, float *adam_m, float *adam_v, int *adam_t,
                    const float *buf_obs, const long long *buf_actions,
                    const float *buf_log_probs, const float *buf_values,
                    const float *buf_valid_masks, const float *buf_rewards,
                    const float *buf_dones, int N,
                    float gamma, float gae_lambda, float clip_eps,
                    float vf_coef, float ent_coef, float lr,
                    float max_grad_norm, int num_epochs, int batch_size,
                    float *out_stats) {
    compute_gae(buf_rewards, buf_values, buf_dones, N, gamma, gae_lambda,
                returns_buf, adv_buf);

    float mean = 0;
    vDSP_meanv(adv_buf, 1, &mean, N);
    /* adv_norm = adv - mean */
    float neg_mean = -mean;
    vDSP_vsadd(adv_buf, 1, &neg_mean, adv_norm_buf, 1, N);
    /* var = dot(adv_norm, adv_norm) / N */
    float var = 0;
    vDSP_dotpr(adv_norm_buf, 1, adv_norm_buf, 1, &var, N);
    var /= N;
    float inv_std = 1.0f / sqrtf(var + 1e-8f);
    vDSP_vsmul(adv_norm_buf, 1, &inv_std, adv_norm_buf, 1, N);

    for (int i = 0; i < N; i++) perm_buf[i] = i;

    float tot_pl=0, tot_vl=0, tot_ent=0, tot_kl=0;
    int num_updates = 0;

    static float *mb_obs_s = NULL;
    static long long *mb_act_s = NULL;
    static float *mb_olp_s = NULL, *mb_ret_s = NULL, *mb_adv_s = NULL;
    static float *mb_vm_s = NULL;
    if (!mb_obs_s) {
        mb_obs_s = (float*)malloc(MAX_BATCH * INPUT_DIM * sizeof(float));
        mb_act_s = (long long*)malloc(MAX_BATCH * sizeof(long long));
        mb_olp_s = (float*)malloc(MAX_BATCH * sizeof(float));
        mb_ret_s = (float*)malloc(MAX_BATCH * sizeof(float));
        mb_adv_s = (float*)malloc(MAX_BATCH * sizeof(float));
        mb_vm_s = (float*)malloc(MAX_BATCH * POLICY_DIM * sizeof(float));
    }

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        shuffle(perm_buf, N);

        for (int start = 0; start < N; start += batch_size) {
            int end = start + batch_size;
            if (end > N) end = N;
            int B = end - start;

            for (int i = 0; i < B; i++) {
                int idx = perm_buf[start + i];
                memcpy(mb_obs_s + i*INPUT_DIM, buf_obs + idx*INPUT_DIM, INPUT_DIM*sizeof(float));
                mb_act_s[i] = buf_actions[idx];
                mb_olp_s[i] = buf_log_probs[idx];
                mb_ret_s[i] = returns_buf[idx];
                mb_adv_s[i] = adv_norm_buf[idx];
                memcpy(mb_vm_s + i*POLICY_DIM, buf_valid_masks + idx*POLICY_DIM, POLICY_DIM*sizeof(float));
            }

            float pl, vl, ent, kl;
            float gnorm_sq = backward_batch(params, mb_obs_s, mb_act_s, mb_olp_s,
                                             mb_ret_s, mb_adv_s, mb_vm_s, B,
                                             clip_eps, vf_coef, ent_coef,
                                             &pl, &vl, &ent, &kl);

            (*adam_t)++;
            adam_step_prenorm(params, grad_buf, adam_m, adam_v, TOTAL_PARAMS,
                              *adam_t, lr, max_grad_norm, gnorm_sq);

            tot_pl += pl; tot_vl += vl; tot_ent += ent; tot_kl += kl;
            num_updates++;
        }
    }

    float inv = 1.0f / (float)(num_updates > 0 ? num_updates : 1);
    out_stats[0] = tot_pl * inv;
    out_stats[1] = tot_vl * inv;
    out_stats[2] = tot_ent * inv;
    out_stats[3] = tot_kl * inv;
}

/* ========== Connect 4 Game Logic ========== */

static int check_win_at(const float *board, int row, int col, float player) {
    static const int dr[4] = {0, 1, 1, 1};
    static const int dc[4] = {1, 0, 1, -1};

    for (int d = 0; d < 4; d++) {
        int count = 1;
        int r = row + dr[d], c = col + dc[d];
        while (r >= 0 && r < BOARD_ROWS && c >= 0 && c < BOARD_COLS
               && board[r*BOARD_COLS + c] == player) {
            count++; r += dr[d]; c += dc[d];
        }
        r = row - dr[d]; c = col - dc[d];
        while (r >= 0 && r < BOARD_ROWS && c >= 0 && c < BOARD_COLS
               && board[r*BOARD_COLS + c] == player) {
            count++; r -= dr[d]; c -= dc[d];
        }
        if (count >= 4) return 1;
    }
    return 0;
}

/* ========== Action Sampling ========== */

static void sample_actions(const float *logits, const float *values,
                            const float *valid_masks, int B,
                            int *out_actions, float *out_lp, float *out_val) {
    for (int i = 0; i < B; i++) {
        const float *lg = logits + i*POLICY_DIM;
        const float *vm = valid_masks + i*POLICY_DIM;

        float masked[POLICY_DIM];
        float mx = -1e30f;
        for (int j = 0; j < POLICY_DIM; j++) {
            masked[j] = vm[j] > 0.5f ? lg[j] : -1e8f;
            if (masked[j] > mx) mx = masked[j];
        }
        float s = 0;
        float p[POLICY_DIM];
        for (int j = 0; j < POLICY_DIM; j++) {
            p[j] = expf(masked[j] - mx);
            s += p[j];
        }
        for (int j = 0; j < POLICY_DIM; j++) p[j] /= s;

        float r = rng_float();
        float cum = 0;
        int action = POLICY_DIM - 1;
        for (int j = 0; j < POLICY_DIM; j++) {
            cum += p[j];
            if (r < cum) { action = j; break; }
        }
        out_actions[i] = action;
        out_lp[i] = logf(p[action] + 1e-8f);
        out_val[i] = values[i];
    }
}

/* ========== Game Collection ========== */

int c4_collect_games(const float *agent_params, const float *opp_params,
                      int num_games, float draw_reward,
                      float *out_obs, long long *out_actions, float *out_log_probs,
                      float *out_values, float *out_valid_masks,
                      float *out_rewards, float *out_dones,
                      int *out_game_results) {
    memset(board_buf, 0, num_games * BOARD_SIZE * sizeof(float));
    memset(col_height_buf, 0, num_games * BOARD_COLS * sizeof(int));
    memset(done_buf_g, 0, num_games * sizeof(int));
    memset(winner_buf, 0, num_games * sizeof(float));
    for (int i = 0; i < num_games; i++)
        cplayer_buf[i] = 1.0f;

    for (int i = 0; i < num_games; i++)
        agent_player_buf[i] = rng_float() < 0.5f ? 1.0f : -1.0f;

    memset(game_trans_count, 0, num_games * sizeof(int));
    int trans_count = 0;
    int games_remaining = num_games;

    int agent_idx[MAX_BATCH], opp_idx[MAX_BATCH];
    int agent_actions[MAX_BATCH], opp_actions[MAX_BATCH];
    float agent_lp[MAX_BATCH], agent_val[MAX_BATCH];

    static float g_obs[MAX_BATCH * INPUT_DIM];
    static float g_vm[MAX_BATCH * POLICY_DIM];
    static float g_logits[MAX_BATCH * POLICY_DIM];
    static float g_values[MAX_BATCH];
    int move_col[MAX_BATCH];

    while (games_remaining > 0) {

        int n_agent = 0, n_opp = 0;
        for (int i = 0; i < num_games; i++) {
            if (done_buf_g[i]) continue;

            float cp = cplayer_buf[i];
            float *ob = obs_buf + i*INPUT_DIM;
            float *bd = board_buf + i*BOARD_SIZE;

            for (int j = 0; j < BOARD_SIZE; j++) {
                ob[j*3]   = (bd[j] == cp)  ? 1.0f : 0.0f;
                ob[j*3+1] = (bd[j] == -cp) ? 1.0f : 0.0f;
                ob[j*3+2] = 1.0f;
            }

            int *ch = col_height_buf + i*BOARD_COLS;
            for (int c = 0; c < BOARD_COLS; c++)
                vmask_buf[i*POLICY_DIM + c] = (ch[c] < BOARD_ROWS) ? 1.0f : 0.0f;

            if (cp == agent_player_buf[i])
                agent_idx[n_agent++] = i;
            else
                opp_idx[n_opp++] = i;
        }

        if (n_agent > 0) {
            for (int i = 0; i < n_agent; i++) {
                memcpy(g_obs + i*INPUT_DIM, obs_buf + agent_idx[i]*INPUT_DIM, INPUT_DIM*sizeof(float));
                memcpy(g_vm + i*POLICY_DIM, vmask_buf + agent_idx[i]*POLICY_DIM, POLICY_DIM*sizeof(float));
            }
            forward_infer(agent_params, g_obs, n_agent, g_logits, g_values);
            sample_actions(g_logits, g_values, g_vm, n_agent,
                           agent_actions, agent_lp, agent_val);

            for (int i = 0; i < n_agent; i++) {
                int gi = agent_idx[i];
                int tc = trans_count++;
                memcpy(tmp_obs + tc*INPUT_DIM, g_obs + i*INPUT_DIM, INPUT_DIM*sizeof(float));
                tmp_actions[tc] = agent_actions[i];
                tmp_log_probs[tc] = agent_lp[i];
                tmp_values[tc] = agent_val[i];
                memcpy(tmp_valid_masks + tc*POLICY_DIM, g_vm + i*POLICY_DIM, POLICY_DIM*sizeof(float));
                game_trans_idx[gi][game_trans_count[gi]++] = tc;

                int col = agent_actions[i];
                int row = col_height_buf[gi*BOARD_COLS + col];
                board_buf[gi*BOARD_SIZE + row*BOARD_COLS + col] = cplayer_buf[gi];
                col_height_buf[gi*BOARD_COLS + col]++;
                move_col[gi] = col;
            }
        }

        if (n_opp > 0) {
            for (int i = 0; i < n_opp; i++) {
                memcpy(g_obs + i*INPUT_DIM, obs_buf + opp_idx[i]*INPUT_DIM, INPUT_DIM*sizeof(float));
                memcpy(g_vm + i*POLICY_DIM, vmask_buf + opp_idx[i]*POLICY_DIM, POLICY_DIM*sizeof(float));
            }
            forward_infer(opp_params, g_obs, n_opp, g_logits, g_values);

            for (int i = 0; i < n_opp; i++) {
                const float *lg = g_logits + i*POLICY_DIM;
                const float *vm = g_vm + i*POLICY_DIM;
                float mx = -1e30f;
                float p[POLICY_DIM], s_val = 0;
                for (int j = 0; j < POLICY_DIM; j++) {
                    float ml = vm[j] > 0.5f ? lg[j] : -1e8f;
                    if (ml > mx) mx = ml;
                    p[j] = ml;
                }
                for (int j = 0; j < POLICY_DIM; j++) { p[j] = expf(p[j]-mx); s_val += p[j]; }
                for (int j = 0; j < POLICY_DIM; j++) p[j] /= s_val;

                float r = rng_float();
                float cum = 0;
                int action = POLICY_DIM - 1;
                for (int j = 0; j < POLICY_DIM; j++) { cum += p[j]; if (r < cum) { action = j; break; } }

                int gi = opp_idx[i];
                int col = action;
                int row = col_height_buf[gi*BOARD_COLS + col];
                board_buf[gi*BOARD_SIZE + row*BOARD_COLS + col] = cplayer_buf[gi];
                col_height_buf[gi*BOARD_COLS + col]++;
                move_col[gi] = col;
            }
        }

        for (int i = 0; i < num_games; i++) {
            if (done_buf_g[i]) continue;
            int col = move_col[i];
            int row = col_height_buf[i*BOARD_COLS + col] - 1;
            float cp = cplayer_buf[i];
            if (check_win_at(board_buf + i*BOARD_SIZE, row, col, cp)) {
                done_buf_g[i] = 1;
                winner_buf[i] = cp;
                games_remaining--;
            } else {
                int full = 1;
                int *ch = col_height_buf + i*BOARD_COLS;
                for (int c = 0; c < BOARD_COLS; c++)
                    if (ch[c] < BOARD_ROWS) { full = 0; break; }
                if (full) {
                    done_buf_g[i] = 1;
                    winner_buf[i] = 0;
                    games_remaining--;
                }
            }
        }

        for (int i = 0; i < num_games; i++)
            if (!done_buf_g[i])
                cplayer_buf[i] *= -1.0f;
    }

    int out_idx = 0;
    int wins = 0, draws = 0, losses = 0;
    memset(out_rewards, 0, trans_count * sizeof(float));
    memset(out_dones, 0, trans_count * sizeof(float));

    for (int g = 0; g < num_games; g++) {
        float ap = agent_player_buf[g];
        float w = winner_buf[g];
        float reward;
        if (w == ap) { reward = 1.0f; wins++; }
        else if (w == 0) { reward = draw_reward; draws++; }
        else { reward = -1.0f; losses++; }

        int n_moves = game_trans_count[g];
        for (int m = 0; m < n_moves; m++) {
            int src = game_trans_idx[g][m];
            memcpy(out_obs + out_idx*INPUT_DIM, tmp_obs + src*INPUT_DIM, INPUT_DIM*sizeof(float));
            out_actions[out_idx] = tmp_actions[src];
            out_log_probs[out_idx] = tmp_log_probs[src];
            out_values[out_idx] = tmp_values[src];
            memcpy(out_valid_masks + out_idx*POLICY_DIM, tmp_valid_masks + src*POLICY_DIM, POLICY_DIM*sizeof(float));

            if (m == n_moves - 1) {
                out_rewards[out_idx] = reward;
                out_dones[out_idx] = 1.0f;
            }
            out_idx++;
        }
    }

    out_game_results[0] = wins;
    out_game_results[1] = draws;
    out_game_results[2] = losses;

    return trans_count;
}
