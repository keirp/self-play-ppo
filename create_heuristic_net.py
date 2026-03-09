"""Train a neural network to imitate the heuristic opponent via supervised learning.
This creates a network that plays like the heuristic, which can be added to the opponent pool.
"""
import numpy as np
import torch
import torch.nn.functional as F
from src.connect4_c import Connect4Net, extract_params
from src.connect4 import Connect4, _heuristic_move

def collect_heuristic_data(num_games=10000):
    """Collect (obs, valid_mask, action) from heuristic playing against random."""
    obs_list = []
    mask_list = []
    action_list = []

    for g in range(num_games):
        env = Connect4()
        obs = env.reset()

        while not env.done:
            valid = env.get_valid_moves()

            # Heuristic plays as current player, collect its decisions
            action = _heuristic_move(env)
            obs_list.append(obs.flatten().copy())
            mask_list.append(valid.copy())
            action_list.append(action)

            obs, _, done, _ = env.step(action)

    return (np.array(obs_list, dtype=np.float32),
            np.array(mask_list, dtype=np.float32),
            np.array(action_list, dtype=np.int64))

def train_imitation(hidden_size=256, num_layers=6, num_epochs=50, lr=1e-3):
    print("Collecting heuristic data...")
    obs, masks, actions = collect_heuristic_data(10000)
    print(f"  {len(obs)} state-action pairs")

    model = Connect4Net(hidden_size=hidden_size, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    obs_t = torch.from_numpy(obs)
    masks_t = torch.from_numpy(masks)
    actions_t = torch.from_numpy(actions)

    dataset = torch.utils.data.TensorDataset(obs_t, masks_t, actions_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_obs, batch_masks, batch_actions in loader:
            logits, _ = model(batch_obs, batch_masks)
            loss = F.cross_entropy(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_obs)
            correct += (logits.argmax(dim=-1) == batch_actions).sum().item()
            total += len(batch_obs)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/total:.4f}, acc={correct/total:.1%}")

    # Test: play vs heuristic
    model.eval()
    policy = model.get_policy_fn("cpu", deterministic=True)
    from src.connect4 import play_vs_heuristic
    w1, d1, l1 = play_vs_heuristic(policy, 1, 50)
    w2, d2, l2 = play_vs_heuristic(policy, -1, 50)
    print(f"  Imitation net vs heuristic: W={w1+w2} D={d1+d2} L={l1+l2}")

    params = extract_params(model)
    return params

if __name__ == "__main__":
    params = train_imitation()
    np.save("data/heuristic_net_params.npy", params)
    print(f"Saved heuristic net params ({len(params)} params)")
