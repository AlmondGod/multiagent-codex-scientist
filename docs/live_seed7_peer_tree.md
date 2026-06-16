# Live Seed 7 Peer-Critique Tree

Run:

```text
runs/live_cultural_lineage_seed7_peer_critique
```

## Tree

```mermaid
flowchart TD
    A0N1["agent_0 node 1<br/>invent: action_grad_dynamics<br/>score 0.957682<br/>val_mse 0.044188"]
    A1N1["agent_1 node 1<br/>invent: smooth_l1_dynamics_pixel<br/>score 0.979632<br/>val_mse 0.020791"]
    A2N1["agent_2 node 1<br/>invent: dynamics_first_schedule<br/>score 0.920532<br/>val_mse 0.086329"]

    A0N2["agent_0 node 2<br/>copy from agent_1 node 1<br/>smooth_l1_dynamics_pixel<br/>score 0.980475<br/>delta +0.022793"]
    A1N2["agent_1 node 2<br/>mutate own node 1<br/>smooth_l1_dynamics_pixel<br/>score 0.980523<br/>delta +0.000891"]
    A2N2["agent_2 node 2<br/>copy from agent_1 node 1<br/>smooth_l1_dynamics_pixel<br/>score 0.980023<br/>delta +0.059491"]

    A0N1 -. "weaker than population best" .-> A0N2
    A1N1 -- "copied by agent_0" --> A0N2
    A1N1 -- "self-mutation" --> A1N2
    A2N1 -. "rejected/abandoned direction" .-> A2N2
    A1N1 -- "copied by agent_2" --> A2N2

    classDef source fill:#d8f5dc,stroke:#2f7d32,color:#111;
    classDef improved fill:#e8f1ff,stroke:#255f99,color:#111;
    classDef weak fill:#ffe8e3,stroke:#a33a2c,color:#111;

    class A1N1 source;
    class A0N2,A1N2,A2N2 improved;
    class A2N1 weak;
```

## Interpretation

Agent 1's first branch was the population-best idea: use
`smooth_l1_dynamics_pixel` with `dynamics_pixel_loss_weight=6.0` and
`motion_prior_weight=1.5`.

At the communication checkpoint:

- Agent 0 copied agent 1's smooth-L1 idea and improved from `0.957682` to `0.980475`.
- Agent 1 mutated its own smooth-L1 idea and improved from `0.979632` to `0.980523`.
- Agent 2 abandoned its weak fast-schedule branch and copied agent 1, improving from `0.920532` to `0.980023`.

This is the clearest single-run tree for the cultural-evolution claim: a strong
idea appears in one branch, then spreads through copy and mutation to improve the
population.
