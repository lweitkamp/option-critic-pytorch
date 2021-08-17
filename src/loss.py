import torch


def critic_loss(model, model_target, optim, data_batch, gamma=0.99):
    optim.zero_grad()

    obs, options, rewards, next_obs, dones = data_batch
    batch_idx = torch.arange(len(options), dtype=torch.long)
    options = torch.LongTensor(options, device=model.device)
    rewards = torch.FloatTensor(rewards, device=model.device)
    dones = torch.FloatTensor(dones, device=model.device)

    # Get outputs of current and next obs, including from target model.
    out = model(obs)
    out_next = model(next_obs)
    out_next_target = model_target(next_obs)

    # Calculate the termination probability of the option in the next state.
    term_prob = out_next['terminations'][batch_idx, options]
    q_next = out_next_target['q']

    # Calculate relative advantage of current option.
    option_advantage = (1 - term_prob) * q_next[batch_idx, options] + \
        term_prob * q_next.max(axis=-1)[0]

    # Calculate the one-step return using the relative advantage.
    gt = rewards + (1 - dones) * gamma * option_advantage

    # Use the return as a target for the temporal-difference error
    td_error = (out['q'][batch_idx, options] - gt.detach()).pow(2).mean()
    td_error.backward()
    optim.step()

    return {'td_error': td_error}


def actor_loss(obs,
               option,
               logp,
               entropy,
               reward,
               done,
               next_obs,
               model,
               model_target,
               optim,
               gamma=0.99, termination_reg=0.1, entropy_reg=0.1):

    optim.zero_grad()

    out = model(obs)
    out_next = model(next_obs)
    out_next_target = model_target(next_obs)

    term_prob_next = out_next['terminations'][option]

    q = out['q'].squeeze()
    q_next_target = out_next_target['q']

    option_advantage = (1 - term_prob_next) * q_next_target[option] + \
        term_prob_next * q_next_target.max(dim=-1)[0]

    gt = reward + (1 - done) * gamma * option_advantage

    policy_loss = -logp * \
        (gt - q[option]).detach() - entropy_reg * entropy

    term_loss = term_prob_next * \
        (q_next_target[option] - q_next_target.max(dim=-1)
         [0] + termination_reg).detach() * (1 - done)

    loss = policy_loss + term_loss
    loss.backward()
    optim.step()
    return {'actor_loss': policy_loss, 'termination_loss': term_loss}
