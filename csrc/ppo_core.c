/*
 * Pure C implementation of PPO training for tic-tac-toe.
 * Uses Apple Accelerate (BLAS) for matrix operations.
 * Eliminates all PyTorch/Python overhead from the training loop.
 */
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Network architecture: 27 -> 256 -> 256 -> 256 -> 256 -> (9 policy, 1 value) */
#define INPUT_DIM 27
#define H 256
#define POLICY_DIM 9
#define VALUE_DIM 1
#define NUM_TRUNK 4

/* Parameter offsets in flat array (matches PyTorch parameter ordering) */
#define OFF_W0 0
#define OFF_B0 (OFF_W0 + H*INPUT_DIM)       /* 6912 */
#define OFF_W1 (OFF_B0 + H)                  /* 7168 */
#define OFF_B1 (OFF_W1 + H*H)                /* 72704 */
#define OFF_W2 (OFF_B1 + H)                  /* 72960 */
#define OFF_B2 (OFF_W2 + H*H)                /* 138496 */
#define OFF_W3 (OFF_B2 + H)                  /* 138752 */
#define OFF_B3 (OFF_W3 + H*H)                /* 204288 */
#define OFF_WP (OFF_B3 + H)                  /* 204544 */
#define OFF_BP (OFF_WP + POLICY_DIM*H)        /* 206848 */
#define OFF_WV (OFF_BP + POLICY_DIM)          /* 206857 */
#define OFF_BV (OFF_WV + VALUE_DIM*H)         /* 207113 */
#define TOTAL_PARAMS (OFF_BV + VALUE_DIM)     /* 207114 */

static const int W_OFF[4] = {OFF_W0, OFF_W1, OFF_W2, OFF_W3};
static const int B_OFF[4] = {OFF_B0, OFF_B1, OFF_B2, OFF_B3};
static const int W_IN[4]  = {INPUT_DIM, H, H, H};

/* Max buffer sizes */
#define MAX_BATCH 1024
#define MAX_TRANS 8192

/* Scratch memory */
static float *act[5];         /* activations: act[0]=h1..act[3]=h4, act[4]=temp */
static float *d_lp_buf;       /* (MAX_BATCH, 9) */
static float *d_logits_buf;   /* (MAX_BATCH, 9) */
static float *d_value_buf;    /* (MAX_BATCH,) */
static float *dh_buf;         /* (MAX_BATCH, H) */
static float *dz_buf;         /* (MAX_BATCH, H) */
static float *grad_buf;       /* TOTAL_PARAMS */
static float *lp_buf;         /* (MAX_BATCH, 9) log_probs */
static float *prob_buf;       /* (MAX_BATCH, 9) probs */
static float *logit_buf;      /* (MAX_BATCH, 9) */
static float *val_buf;        /* (MAX_BATCH,) */
static float *returns_buf;    /* MAX_TRANS */
static float *adv_buf;        /* MAX_TRANS */
static float *adv_norm_buf;   /* MAX_TRANS */
static int   *perm_buf;       /* MAX_TRANS */
/* Epoch-level activation storage (for batched forward) */
static float *epoch_act[4];  /* Each MAX_TRANS * H */
static float *epoch_logits;  /* MAX_TRANS * POLICY_DIM */
static float *epoch_values;  /* MAX_TRANS */
static float *epoch_lp;      /* MAX_TRANS * POLICY_DIM */
static float *epoch_probs;   /* MAX_TRANS * POLICY_DIM */
static float *ones_vec;      /* MAX_BATCH for bias grad */

static int inited = 0;

/* ---- Board game scratch ---- */
static float *board_buf;      /* MAX_BATCH * 9 */
static float *obs_buf;        /* MAX_BATCH * 27 */
static float *vmask_buf;      /* MAX_BATCH * 9 */
static float *cplayer_buf;    /* MAX_BATCH */
static int   *done_buf_g;     /* MAX_BATCH */
static float *winner_buf;     /* MAX_BATCH */
static float *agent_player_buf; /* MAX_BATCH */

void ppo_init(void) {
    if (inited) return;
    for (int i = 0; i < 5; i++)
        act[i] = (float*)calloc(MAX_BATCH * H, sizeof(float));
    d_lp_buf = (float*)calloc(MAX_BATCH * POLICY_DIM, sizeof(float));
    d_logits_buf = (float*)calloc(MAX_BATCH * POLICY_DIM, sizeof(float));
    d_value_buf = (float*)calloc(MAX_BATCH, sizeof(float));
    dh_buf = (float*)calloc(MAX_BATCH * H, sizeof(float));
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
    board_buf = (float*)calloc(MAX_BATCH * 9, sizeof(float));
    obs_buf = (float*)calloc(MAX_BATCH * 27, sizeof(float));
    vmask_buf = (float*)calloc(MAX_BATCH * 9, sizeof(float));
    cplayer_buf = (float*)calloc(MAX_BATCH, sizeof(float));
    done_buf_g = (int*)calloc(MAX_BATCH, sizeof(int));
    winner_buf = (float*)calloc(MAX_BATCH, sizeof(float));
    agent_player_buf = (float*)calloc(MAX_BATCH, sizeof(float));
    for (int i = 0; i < 4; i++)
        epoch_act[i] = (float*)calloc(MAX_TRANS * H, sizeof(float));
    epoch_logits = (float*)calloc(MAX_TRANS * POLICY_DIM, sizeof(float));
    epoch_values = (float*)calloc(MAX_TRANS, sizeof(float));
    epoch_lp = (float*)calloc(MAX_TRANS * POLICY_DIM, sizeof(float));
    epoch_probs = (float*)calloc(MAX_TRANS * POLICY_DIM, sizeof(float));
    ones_vec = (float*)calloc(MAX_BATCH, sizeof(float));
    for (int i = 0; i < MAX_BATCH; i++) ones_vec[i] = 1.0f;
    inited = 1;
}

/* xoshiro128+ RNG - much better quality than drand48 */
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
    return (float)(xoshiro128p() >> 8) / 16777216.0f;  /* 24-bit mantissa */
}

void ppo_seed(unsigned long seed) {
    /* SplitMix64 to initialize xoshiro state from a single seed */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        rng_s[i] = (uint32_t)z;
    }
}

/* ========== Core math operations ========== */

/* out = in @ W^T + b,  in:(B,K) W:(N,K) b:(N) out:(B,N) */
static inline void linear_fwd(const float *in, const float *W, const float *b,
                               float *out, int B, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, N, K, 1.0f, in, K, W, K, 0.0f, out, N);
    for (int i = 0; i < B; i++)
        vDSP_vadd(out + i*N, 1, b, 1, out + i*N, 1, N);
}


/* ReLU in-place */
static inline void relu_ip(float *x, int n) {
    float zero = 0.0f;
    vDSP_vthres(x, 1, &zero, x, 1, n);
}

/* ========== Forward pass ========== */

/* Forward with activation saving (for training backward pass).
   Saves activations in act[0..3]. Input obs is kept by pointer. */
static void forward_train(const float *params, const float *obs, int B,
                           float *out_logits, float *out_values) {
    /* Layer 0: act[0] = relu(obs @ W0^T + b0) */
    linear_fwd(obs, params+OFF_W0, params+OFF_B0, act[0], B, INPUT_DIM, H);
    relu_ip(act[0], B*H);

    /* Layers 1-3 */
    for (int l = 1; l < NUM_TRUNK; l++) {
        linear_fwd(act[l-1], params+W_OFF[l], params+B_OFF[l], act[l], B, H, H);
        relu_ip(act[l], B*H);
    }

    /* Policy head: logits = act[3] @ Wp^T + bp */
    linear_fwd(act[3], params+OFF_WP, params+OFF_BP, out_logits, B, H, POLICY_DIM);

    /* Value head: values = act[3] @ Wv^T + bv */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, VALUE_DIM, H, 1.0f, act[3], H, params+OFF_WV, H,
                0.0f, out_values, VALUE_DIM);
    float bv = params[OFF_BV];
    for (int i = 0; i < B; i++) out_values[i] += bv;
}

/* Forward inference only (no activation saving needed for game collection).
   Uses act[4] as temp, act[0] and act[1] as ping-pong buffers. */
static void forward_infer(const float *params, const float *obs, int B,
                            float *out_logits, float *out_values) {
    float *cur = act[0], *nxt = act[1];

    linear_fwd(obs, params+OFF_W0, params+OFF_B0, cur, B, INPUT_DIM, H);
    relu_ip(cur, B*H);

    for (int l = 1; l < NUM_TRUNK; l++) {
        linear_fwd(cur, params+W_OFF[l], params+B_OFF[l], nxt, B, H, H);
        relu_ip(nxt, B*H);
        float *tmp = cur; cur = nxt; nxt = tmp;
    }
    /* cur now points to final hidden activation */
    linear_fwd(cur, params+OFF_WP, params+OFF_BP, out_logits, B, H, POLICY_DIM);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, VALUE_DIM, H, 1.0f, cur, H, params+OFF_WV, H,
                0.0f, out_values, VALUE_DIM);
    float bv = params[OFF_BV];
    for (int i = 0; i < B; i++) out_values[i] += bv;
}

/* Forward pass saving to epoch-level buffers (for all N samples at once). */
static void forward_epoch(const float *params, const float *obs, int N) {
    linear_fwd(obs, params+OFF_W0, params+OFF_B0, epoch_act[0], N, INPUT_DIM, H);
    relu_ip(epoch_act[0], N*H);
    for (int l = 1; l < NUM_TRUNK; l++) {
        linear_fwd(epoch_act[l-1], params+W_OFF[l], params+B_OFF[l], epoch_act[l], N, H, H);
        relu_ip(epoch_act[l], N*H);
    }
    linear_fwd(epoch_act[3], params+OFF_WP, params+OFF_BP, epoch_logits, N, H, POLICY_DIM);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, VALUE_DIM, H, 1.0f, epoch_act[3], H, params+OFF_WV, H,
                0.0f, epoch_values, VALUE_DIM);
    float bv = params[OFF_BV];
    for (int i = 0; i < N; i++) epoch_values[i] += bv;
}

/* Backward-only: uses pre-computed epoch activations for given mini-batch indices.
   Does NOT do forward pass. Uses epoch_act, epoch_lp, epoch_probs for activations. */
static void backward_batch_precomputed(
    const float *params, const int *indices, int B,
    const float *obs_all, const long long *actions_all,
    const float *old_lp_all, const float *ret_all, const float *adv_all,
    const float *valid_masks_all,
    float clip_eps, float vf_coef, float ent_coef,
    float *out_pl, float *out_vl, float *out_ent, float *out_kl) {

    float inv_B = 1.0f / (float)B;
    float policy_loss = 0, value_loss = 0, entropy = 0, approx_kl = 0;
    memset(d_lp_buf, 0, B * POLICY_DIM * sizeof(float));

    /* Extract mini-batch activations into act[0..3] for backward */
    for (int i = 0; i < B; i++) {
        int idx = indices[i];
        memcpy(act[0] + i*H, epoch_act[0] + idx*H, H*sizeof(float));
        memcpy(act[1] + i*H, epoch_act[1] + idx*H, H*sizeof(float));
        memcpy(act[2] + i*H, epoch_act[2] + idx*H, H*sizeof(float));
        memcpy(act[3] + i*H, epoch_act[3] + idx*H, H*sizeof(float));
    }

    /* Use pre-computed logits/probs/lp for loss computation */
    for (int i = 0; i < B; i++) {
        int idx = indices[i];
        int ai = (int)actions_all[idx];
        float new_lpi = epoch_lp[idx*POLICY_DIM + ai];
        float ratio = expf(new_lpi - old_lp_all[idx]);

        float surr1 = ratio * adv_all[i];  /* adv_all here is already extracted/normalized */
        float clipped = ratio < 1-clip_eps ? 1-clip_eps : (ratio > 1+clip_eps ? 1+clip_eps : ratio);
        float surr2 = clipped * adv_all[i];
        float s = surr1 < surr2 ? surr1 : surr2;
        policy_loss -= s * inv_B;

        float d_new_lp;
        if (surr1 <= surr2)
            d_new_lp = -adv_all[i] * ratio * inv_B;
        else
            d_new_lp = (clipped == ratio) ? -adv_all[i] * ratio * inv_B : 0.0f;
        d_lp_buf[i*POLICY_DIM + ai] += d_new_lp;

        float vdiff = epoch_values[idx] - ret_all[i];
        value_loss += vdiff * vdiff * inv_B;
        d_value_buf[i] = vf_coef * 2.0f * vdiff * inv_B;

        float ent_i = 0;
        for (int j = 0; j < POLICY_DIM; j++)
            ent_i -= epoch_probs[idx*POLICY_DIM+j] * epoch_lp[idx*POLICY_DIM+j];
        entropy += ent_i * inv_B;

        for (int j = 0; j < POLICY_DIM; j++)
            d_lp_buf[i*POLICY_DIM+j] += ent_coef * inv_B
                * epoch_probs[idx*POLICY_DIM+j] * (epoch_lp[idx*POLICY_DIM+j] + 1.0f);

        approx_kl += (old_lp_all[idx] - new_lpi) * inv_B;
    }

    *out_pl = policy_loss; *out_vl = value_loss; *out_ent = entropy; *out_kl = approx_kl;

    /* Log-softmax backward */
    for (int i = 0; i < B; i++) {
        float sum_dlp = 0;
        int idx = indices[i];
        for (int j = 0; j < POLICY_DIM; j++) sum_dlp += d_lp_buf[i*POLICY_DIM+j];
        for (int j = 0; j < POLICY_DIM; j++)
            d_logits_buf[i*POLICY_DIM+j] = d_lp_buf[i*POLICY_DIM+j]
                - epoch_probs[idx*POLICY_DIM+j] * sum_dlp;
    }

    /* Extract mini-batch obs for weight gradient */
    static float mb_obs_bw[MAX_BATCH * INPUT_DIM];
    for (int i = 0; i < B; i++)
        memcpy(mb_obs_bw + i*INPUT_DIM, obs_all + indices[i]*INPUT_DIM, INPUT_DIM*sizeof(float));

    /* Backward through heads */
    float *gWp = grad_buf + OFF_WP, *gbp = grad_buf + OFF_BP;
    float *gWv = grad_buf + OFF_WV, *gbv = grad_buf + OFF_BV;

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                POLICY_DIM, H, B, 1.0f, d_logits_buf, POLICY_DIM, act[3], H, 1.0f, gWp, H);
    cblas_sgemv(CblasRowMajor, CblasTrans, B, POLICY_DIM,
                1.0f, d_logits_buf, POLICY_DIM, ones_vec, 1, 1.0f, gbp, 1);

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                VALUE_DIM, H, B, 1.0f, d_value_buf, VALUE_DIM, act[3], H, 1.0f, gWv, H);
    for (int i = 0; i < B; i++) gbv[0] += d_value_buf[i];

    /* dh = d_logits @ Wp + d_value outer Wv */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                B, H, POLICY_DIM, 1.0f, d_logits_buf, POLICY_DIM, params+OFF_WP, H, 0.0f, dh_buf, H);
    for (int i = 0; i < B; i++) {
        float dv = d_value_buf[i];
        const float *wv = params + OFF_WV;
        for (int j = 0; j < H; j++) dh_buf[i*H+j] += dv * wv[j];
    }

    /* Backprop through trunk */
    for (int l = NUM_TRUNK-1; l >= 0; l--) {
        for (int k = 0; k < B*H; k++)
            dz_buf[k] = act[l][k] > 0.0f ? dh_buf[k] : 0.0f;

        const float *h_prev = (l == 0) ? mb_obs_bw : act[l-1];
        int in_dim = W_IN[l];
        float *gW = grad_buf + W_OFF[l], *gb = grad_buf + B_OFF[l];

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    H, in_dim, B, 1.0f, dz_buf, H, h_prev, in_dim, 1.0f, gW, in_dim);
        cblas_sgemv(CblasRowMajor, CblasTrans, B, H,
                    1.0f, dz_buf, H, ones_vec, 1, 1.0f, gb, 1);

        if (l > 0)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        B, H, H, 1.0f, dz_buf, H, params+W_OFF[l], H, 0.0f, dh_buf, H);
    }
}

/* ========== Log-softmax and loss computation ========== */

/* Compute log_softmax in-place: logits -> log_probs, also write probs.
   Applies valid_mask: adds -1e8 to invalid positions first. */
static void log_softmax_masked(float *logits, const float *valid_masks,
                                 float *log_probs, float *probs, int B) {
    for (int i = 0; i < B; i++) {
        float *lg = logits + i*POLICY_DIM;
        const float *vm = valid_masks + i*POLICY_DIM;
        float *lp = log_probs + i*POLICY_DIM;
        float *pr = probs + i*POLICY_DIM;

        /* Apply mask (branch to avoid -ffast-math cancellation) */
        for (int j = 0; j < POLICY_DIM; j++)
            if (vm[j] < 0.5f) lg[j] = -1e8f;

        /* Find max for numerical stability */
        float mx = lg[0];
        for (int j = 1; j < POLICY_DIM; j++)
            if (lg[j] > mx) mx = lg[j];

        /* exp and sum */
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

/* Helper: compute dot product of a segment */
static inline float segment_norm_sq(const float *x, int n) {
    float s = 0;
    vDSP_dotpr(x, 1, x, 1, &s, n);
    return s;
}

/* ========== Backward pass ========== */

/* Compute gradients for one mini-batch.
   obs: (B,27), actions: (B,), old_lp: (B,), returns: (B,), advantages: (B,)
   valid_masks: (B,9).
   Writes gradients to grad_buf (accumulated, caller must zero first).
   Uses saved activations in act[0..3]. */
static float backward_batch(const float *params, const float *obs,
                            const long long *actions, const float *old_lp,
                            const float *ret, const float *adv,
                            const float *valid_masks,
                            int B, float clip_eps, float vf_coef, float ent_coef,
                            float *out_pl, float *out_vl, float *out_ent, float *out_kl) {
    /* Forward pass (saves activations) */
    forward_train(params, obs, B, logit_buf, val_buf);

    /* Log-softmax */
    log_softmax_masked(logit_buf, valid_masks, lp_buf, prob_buf, B);

    /* Gather new log probs and compute ratio */
    float policy_loss = 0.0f, value_loss = 0.0f, entropy = 0.0f, approx_kl = 0.0f;
    float inv_B = 1.0f / (float)B;

    /* Zero d_lp */
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

        /* d_new_lp from policy */
        float d_new_lp;
        if (surr1 <= surr2) {
            d_new_lp = -adv[i] * ratio * inv_B;
        } else {
            d_new_lp = (clipped == ratio) ? -adv[i] * ratio * inv_B : 0.0f;
        }
        d_lp_buf[i*POLICY_DIM + ai] += d_new_lp;

        /* Value loss */
        float vdiff = val_buf[i] - ret[i];
        value_loss += vdiff * vdiff * inv_B;
        d_value_buf[i] = vf_coef * 2.0f * vdiff * inv_B;

        /* Entropy: H = -sum p*lp */
        float ent_i = 0.0f;
        for (int j = 0; j < POLICY_DIM; j++)
            ent_i -= prob_buf[i*POLICY_DIM+j] * lp_buf[i*POLICY_DIM+j];
        entropy += ent_i * inv_B;

        /* Entropy gradient w.r.t. log_probs */
        for (int j = 0; j < POLICY_DIM; j++) {
            d_lp_buf[i*POLICY_DIM+j] += ent_coef * inv_B * prob_buf[i*POLICY_DIM+j]
                                          * (lp_buf[i*POLICY_DIM+j] + 1.0f);
        }

        approx_kl += (old_lp[i] - new_lpi) * inv_B;
    }

    *out_pl = policy_loss;
    *out_vl = value_loss;
    *out_ent = entropy;
    *out_kl = approx_kl;
    /* Log-softmax backward: d_logits[i,j] = d_lp[i,j] - p[i,j] * sum_k(d_lp[i,k]) */
    for (int i = 0; i < B; i++) {
        float sum_dlp = 0.0f;
        for (int j = 0; j < POLICY_DIM; j++)
            sum_dlp += d_lp_buf[i*POLICY_DIM+j];
        for (int j = 0; j < POLICY_DIM; j++)
            d_logits_buf[i*POLICY_DIM+j] = d_lp_buf[i*POLICY_DIM+j]
                                            - prob_buf[i*POLICY_DIM+j] * sum_dlp;
    }

    /* ---- Backward through network ---- */
    /* NOTE: Using beta=0 instead of 1.0 to skip the memset(grad_buf,0) */
    /* Gradient for policy head: dWp = d_logits^T @ act[3], dbp = sum(d_logits) */
    float *gWp = grad_buf + OFF_WP;
    float *gbp = grad_buf + OFF_BP;
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                POLICY_DIM, H, B, 1.0f, d_logits_buf, POLICY_DIM, act[3], H,
                0.0f, gWp, H);
    cblas_sgemv(CblasRowMajor, CblasTrans, B, POLICY_DIM,
                1.0f, d_logits_buf, POLICY_DIM, ones_vec, 1, 0.0f, gbp, 1);

    /* Gradient for value head: dWv = d_value^T @ act[3], dbv = sum(d_value) */
    float *gWv = grad_buf + OFF_WV;
    float *gbv = grad_buf + OFF_BV;
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                VALUE_DIM, H, B, 1.0f, d_value_buf, VALUE_DIM, act[3], H,
                0.0f, gWv, H);
    {
        float dv_sum = 0;
        for (int i = 0; i < B; i++) dv_sum += d_value_buf[i];
        gbv[0] = dv_sum;
    }
    /* dh = d_logits @ Wp + d_value ⊗ Wv -> dh_buf (B, H) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                B, H, POLICY_DIM, 1.0f, d_logits_buf, POLICY_DIM, params+OFF_WP, H,
                0.0f, dh_buf, H);
    /* Value head contribution: dh += d_value * Wv (rank-1 update) */
    cblas_sger(CblasRowMajor, B, H, 1.0f, d_value_buf, 1, params+OFF_WV, 1, dh_buf, H);

    /* Backprop through trunk layers 3..0 */
    for (int l = NUM_TRUNK-1; l >= 0; l--) {
        /* ReLU backward: dz = dh * (act[l] > 0) */
        for (int k = 0; k < B*H; k++)
            dz_buf[k] = act[l][k] > 0.0f ? dh_buf[k] : 0.0f;

        /* Weight gradient: dW = dz^T @ h_prev (beta=0: overwrite, no memset needed) */
        const float *h_prev = (l == 0) ? obs : act[l-1];
        int in_dim = W_IN[l];
        float *gW = grad_buf + W_OFF[l];
        float *gb = grad_buf + B_OFF[l];

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    H, in_dim, B, 1.0f, dz_buf, H, h_prev, in_dim,
                    0.0f, gW, in_dim);

        /* Bias gradient: column sums of dz_buf (B x H) */
        cblas_sgemv(CblasRowMajor, CblasTrans, B, H,
                    1.0f, dz_buf, H, ones_vec, 1, 0.0f, gb, 1);

        /* Propagate to previous layer (skip for l=0) */
        if (l > 0) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        B, H, H, 1.0f, dz_buf, H, params+W_OFF[l], H,
                        0.0f, dh_buf, H);
        }
    }
    /* Compute grad norm squared from segments (data still in cache from GEMM output) */
    float gnorm_sq = 0;
    /* Trunk weights + biases */
    for (int l = 0; l < NUM_TRUNK; l++) {
        gnorm_sq += segment_norm_sq(grad_buf + W_OFF[l], H * W_IN[l]);
        gnorm_sq += segment_norm_sq(grad_buf + B_OFF[l], H);
    }
    /* Head weights + biases */
    gnorm_sq += segment_norm_sq(grad_buf + OFF_WP, POLICY_DIM * H);
    gnorm_sq += segment_norm_sq(grad_buf + OFF_BP, POLICY_DIM);
    gnorm_sq += segment_norm_sq(grad_buf + OFF_WV, VALUE_DIM * H);
    gnorm_sq += segment_norm_sq(grad_buf + OFF_BV, VALUE_DIM);
    return gnorm_sq;
}

/* ========== Adam optimizer (grad norm pre-computed) ========== */
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



/* ========== GAE computation ========== */
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

/* ========== Fisher-Yates shuffle ========== */
static void shuffle(int *arr, int n) {
    for (int i = n-1; i > 0; i--) {
        int j = (int)(rng_float() * (i+1));
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/* ========== Full PPO Update ========== */
void ppo_update(float *params, float *adam_m, float *adam_v, int *adam_t,
                const float *buf_obs, const long long *buf_actions,
                const float *buf_log_probs, const float *buf_values,
                const float *buf_valid_masks, const float *buf_rewards,
                const float *buf_dones, int N,
                float gamma, float gae_lambda, float clip_eps,
                float vf_coef, float ent_coef, float lr,
                float max_grad_norm, int num_epochs, int batch_size,
                float *out_stats) {
    /* GAE */
    compute_gae(buf_rewards, buf_values, buf_dones, N, gamma, gae_lambda,
                returns_buf, adv_buf);

    /* Normalize advantages */
    float mean = 0, var = 0;
    for (int i = 0; i < N; i++) mean += adv_buf[i];
    mean /= N;
    for (int i = 0; i < N; i++) {
        float d = adv_buf[i] - mean;
        var += d*d;
    }
    float std = sqrtf(var / N + 1e-8f);
    for (int i = 0; i < N; i++)
        adv_norm_buf[i] = (adv_buf[i] - mean) / std;

    /* Init permutation */
    for (int i = 0; i < N; i++) perm_buf[i] = i;

    float tot_pl=0, tot_vl=0, tot_ent=0, tot_kl=0;
    int num_updates = 0;

    /* Mini-batch extraction buffers */
    static float mb_obs_s[MAX_BATCH * INPUT_DIM];
    static long long mb_act_s[MAX_BATCH];
    static float mb_olp_s[MAX_BATCH];
    static float mb_ret_s[MAX_BATCH];
    static float mb_adv_s[MAX_BATCH];
    static float mb_vm_s[MAX_BATCH * POLICY_DIM];

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        shuffle(perm_buf, N);

        for (int start = 0; start < N; start += batch_size) {
            int end = start + batch_size;
            if (end > N) end = N;
            int B = end - start;

            /* Extract mini-batch */
            for (int i = 0; i < B; i++) {
                int idx = perm_buf[start + i];
                memcpy(mb_obs_s + i*INPUT_DIM, buf_obs + idx*INPUT_DIM, INPUT_DIM*sizeof(float));
                mb_act_s[i] = buf_actions[idx];
                mb_olp_s[i] = buf_log_probs[idx];
                mb_ret_s[i] = returns_buf[idx];
                mb_adv_s[i] = adv_norm_buf[idx];
                memcpy(mb_vm_s + i*POLICY_DIM, buf_valid_masks + idx*POLICY_DIM, POLICY_DIM*sizeof(float));
            }
            /* Forward + backward */
            float pl, vl, ent, kl;
            float gnorm_sq = backward_batch(params, mb_obs_s, mb_act_s, mb_olp_s, mb_ret_s, mb_adv_s,
                           mb_vm_s, B, clip_eps, vf_coef, ent_coef,
                           &pl, &vl, &ent, &kl);

            /* Adam step (grad norm already computed in backward) */
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

/* ========== Tic-Tac-Toe Game Logic ========== */

static const int WIN_LINES[8][3] = {
    {0,1,2},{3,4,5},{6,7,8},
    {0,3,6},{1,4,7},{2,5,8},
    {0,4,8},{2,4,6}
};

static int check_winner_single(const float *board) {
    for (int w = 0; w < 8; w++) {
        float a = board[WIN_LINES[w][0]];
        float b = board[WIN_LINES[w][1]];
        float c = board[WIN_LINES[w][2]];
        if (a != 0 && a == b && b == c) return (int)a;
    }
    return 0;
}

static int board_full(const float *board) {
    for (int i = 0; i < 9; i++)
        if (board[i] == 0) return 0;
    return 1;
}

/* ========== Game Collection ========== */

/* Softmax + multinomial sample for inference.
   logits: (B, 9), valid_masks: (B, 9).
   Writes actions to out_actions, log_probs to out_lp, values to out_val. */
static void sample_actions(const float *logits, const float *values,
                            const float *valid_masks, int B,
                            int *out_actions, float *out_lp, float *out_val) {
    for (int i = 0; i < B; i++) {
        const float *lg = logits + i*POLICY_DIM;
        const float *vm = valid_masks + i*POLICY_DIM;

        /* Masked softmax (branch to avoid -ffast-math cancellation) */
        float masked[9];
        float mx = -1e30f;
        for (int j = 0; j < 9; j++) {
            masked[j] = vm[j] > 0.5f ? lg[j] : -1e8f;
            if (masked[j] > mx) mx = masked[j];
        }
        float s = 0;
        float p[9];
        for (int j = 0; j < 9; j++) {
            p[j] = expf(masked[j] - mx);
            s += p[j];
        }
        for (int j = 0; j < 9; j++) p[j] /= s;

        /* Multinomial sample */
        float r = rng_float();
        float cum = 0;
        int action = 8;
        for (int j = 0; j < 9; j++) {
            cum += p[j];
            if (r < cum) { action = j; break; }
        }
        out_actions[i] = action;
        out_lp[i] = logf(p[action] + 1e-8f);
        out_val[i] = values[i];
    }
}

/* Collect self-play games entirely in C.
   agent_params, opp_params: flat parameter arrays.
   Writes buffer data to output arrays. Returns number of transitions.
   Transitions are reordered so each game's moves are consecutive (required for GAE). */
int collect_games(const float *agent_params, const float *opp_params,
                   int num_games, float draw_reward,
                   float *out_obs, long long *out_actions, float *out_log_probs,
                   float *out_values, float *out_valid_masks,
                   float *out_rewards, float *out_dones,
                   int *out_game_results) {
    /* Initialize games */
    memset(board_buf, 0, num_games * 9 * sizeof(float));
    memset(done_buf_g, 0, num_games * sizeof(int));
    memset(winner_buf, 0, num_games * sizeof(float));
    for (int i = 0; i < num_games; i++)
        cplayer_buf[i] = 1.0f;

    /* Random agent side */
    for (int i = 0; i < num_games; i++)
        agent_player_buf[i] = rng_float() < 0.5f ? 1.0f : -1.0f;

    /* Per-game transition tracking: game_trans_idx[game][move] = buffer position */
    static int game_trans_count[MAX_BATCH];
    static int game_trans_idx[MAX_BATCH][10]; /* max 5 agent moves per game */
    memset(game_trans_count, 0, num_games * sizeof(int));

    int trans_count = 0;
    int last_trans[MAX_BATCH];
    for (int i = 0; i < num_games; i++) last_trans[i] = -1;

    /* Temp buffers for indices */
    int agent_idx[MAX_BATCH], opp_idx[MAX_BATCH];
    int agent_actions[MAX_BATCH], opp_actions[MAX_BATCH];
    float agent_lp[MAX_BATCH], agent_val[MAX_BATCH];

    /* Temp buffer for interleaved collection (will be reordered later) */
    static float tmp_obs[MAX_TRANS * INPUT_DIM];
    static long long tmp_actions[MAX_TRANS];
    static float tmp_log_probs[MAX_TRANS];
    static float tmp_values[MAX_TRANS];
    static float tmp_valid_masks[MAX_TRANS * POLICY_DIM];

    while (1) {
        /* Check if all done */
        int all_done = 1;
        for (int i = 0; i < num_games; i++)
            if (!done_buf_g[i]) { all_done = 0; break; }
        if (all_done) break;

        /* Build obs and valid masks for all active games */
        int n_agent = 0, n_opp = 0;
        for (int i = 0; i < num_games; i++) {
            if (done_buf_g[i]) continue;

            /* Compute observation */
            float cp = cplayer_buf[i];
            float *ob = obs_buf + i*INPUT_DIM;
            float *bd = board_buf + i*9;
            for (int j = 0; j < 9; j++) {
                ob[j*3]   = (bd[j] == cp)  ? 1.0f : 0.0f;
                ob[j*3+1] = (bd[j] == -cp) ? 1.0f : 0.0f;
                ob[j*3+2] = 1.0f;
            }
            /* Valid moves */
            for (int j = 0; j < 9; j++)
                vmask_buf[i*9+j] = (bd[j] == 0) ? 1.0f : 0.0f;

            if (cp == agent_player_buf[i])
                agent_idx[n_agent++] = i;
            else
                opp_idx[n_opp++] = i;
        }

        /* Static buffers for gathered observations/masks */
        static float g_obs[MAX_BATCH * INPUT_DIM];
        static float g_vm[MAX_BATCH * POLICY_DIM];
        static float g_logits[MAX_BATCH * POLICY_DIM];
        static float g_values[MAX_BATCH];

        /* Agent inference */
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
                last_trans[gi] = tc;
                board_buf[gi*9 + agent_actions[i]] = cplayer_buf[gi];
            }
        }

        /* Opponent inference */
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
                float p[9], s = 0;
                for (int j = 0; j < 9; j++) {
                    float ml = vm[j] > 0.5f ? lg[j] : -1e8f;
                    if (ml > mx) mx = ml;
                    p[j] = ml;
                }
                for (int j = 0; j < 9; j++) { p[j] = expf(p[j]-mx); s += p[j]; }
                for (int j = 0; j < 9; j++) p[j] /= s;

                float r = rng_float();
                float cum = 0;
                int action = 8;
                for (int j = 0; j < 9; j++) { cum += p[j]; if (r < cum) { action = j; break; } }

                int gi = opp_idx[i];
                board_buf[gi*9 + action] = cplayer_buf[gi];
            }
        }

        /* Check wins and draws for all active games that had moves */
        for (int i = 0; i < num_games; i++) {
            if (done_buf_g[i]) continue;
            int w = check_winner_single(board_buf + i*9);
            if (w != 0) {
                done_buf_g[i] = 1;
                winner_buf[i] = (float)w;
            } else if (board_full(board_buf + i*9)) {
                done_buf_g[i] = 1;
                winner_buf[i] = 0;
            } else {
                cplayer_buf[i] *= -1.0f;
            }
        }
    }

    /* Reorder transitions: group by game so each game's moves are consecutive */
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
