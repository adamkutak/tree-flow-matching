# Rebuttal Workspace: Inference-Time Compute Scaling for Flow Matching

## Completed Tasks:
- ✅ Ablation over branching schedule (Appendix C.1)
- ✅ Non-uniform branching experiment (Appendix C.2)
- ✅ Classifier-free guidance ablation (Appendix C.3)
- ✅ CFG-based branching ablation (Appendix C.4)
- ✅ Coarse simulate-forward ablation with dt=0.05 (Appendix C.5)
- ✅ Coarse simulate-forward ablation with dt=0.01 (Appendix C.5)
- ✅ Updated divergence-free proof to use Fokker-Planck equation (Section 4.1)

## Pending Tasks:
- ⏳ FLUX ablations (will be completed for final version)




# Global Response to Area Chair and Reviewers

Dear AC, SAC, and PCs,

Thank you for your time and effort in evaluating our work. We believe that this review process has substantially improved the quality of our manuscript, strengthened our contributions, and further clarified the novelty of our work. We were encouraged to see that reviewers recognized the "timely and relevant" nature of the problem **[WCen]**, finding our setting "well-motivated" **[kwmg, AfW2]** and "conceptually straightforward yet effective" **[gTSm]**. Reviewers specifically highlighted that our method is the "first to propose inference-time compute scaling tailored to flow matching while preserving the linear interpolant" **[AfW2]**, avoiding the need for diffusion-style VP conversion. Furthermore, reviewers appreciated that we demonstrated efficacy not just in vision, but also on "unconditional protein design... showing that the approach is not limited to vision and can benefit scientific domains" **[AfW2]**.

Below, we provide a summary of our key novelty claims, the substantial new experiments and theoretical contributions added during the review period, and how we addressed all reviewers' concerns. We provided an updated manuscript (revisions in blue text) incorporating all suggested changes, including six new ablation studies in the appendix and one new experiment (CFG-based branching) added to the main text.

### Summary of Novelty

Our work introduces several distinct contributions to the field of inference-time compute scaling for generative models:

1. **First inference-time scaling method for Flow Matching preserving the linear interpolant.** While Kim et al. (2025) address FM scaling by converting to a VP-interpolant (effectively treating the model as a diffusion model), our Noise Search algorithm is the first to enable inference scaling while strictly adhering to the flow matching ODE paradigm. This preserves FM's key advantages: straighter trajectories, fewer sampling steps, and compatibility with non-Gaussian prior distributions.

2. **First application of inference-time scaling to scientific domains (protein design).** Prior work on inference-time scaling for stochastic interpolant models has been limited to image generation. We demonstrate substantial improvements in protein designability metrics using FoldFlow2, opening inference-time scaling to the broader scientific community where flow matching has seen significant adoption.

3. **CFG-based branching: A novel noise-free branching mechanism (Section 5.2, NEW).** We introduce an alternative to stochastic noise injection that creates diverse branching trajectories by varying the Classifier-Free Guidance (CFG) scale at each branching point. **This is a significant contribution**: it enables inference-time compute scaling while retaining the exact deterministic ODE, without any stochasticity. This approach works for any conditional flow model using CFG and represents a fundamentally different paradigm for trajectory diversification.

4. **Theoretical grounding via Fokker-Planck equation (Section 4.1, NEW).** We provide a rigorous derivation showing that score-orthogonal perturbations are divergence-free with respect to the drift term in the Fokker-Planck equation, addressing reviewer concerns about the theoretical specification of our noise injection method.

5. **Two-stage algorithm exploiting FM's prior distribution invariance.** Our RS+NS algorithm leverages a unique property of flow matching—invariance to the initial distribution—to independently optimize over initial conditions and trajectories, achieving state-of-the-art results.

### New Experiments and Ablations

Based on reviewer feedback, we have added **six comprehensive ablation studies** that both validate our method's robustness and introduce efficiency improvements that further increase our novelty:

| Ablation | Key Finding | Novelty/Efficiency Improvement |
|----------|-------------|-------------------------------|
| **Branching Schedule (C.1)** | Performance remains stable across schedules from ~3.55 to ~1.55 trajectory equivalents | Demonstrates we can use **60% less compute** with tighter schedules while still outperforming random search |
| **Non-Uniform Branching (C.2)** | Uniform branching performs best; non-uniform schedules do not improve performance | Validates default design choice |
| **CFG Compatibility (C.3)** | Our method is fully compatible with CFG sampling | Broader applicability |
| **CFG-Based Branching (C.4, Main Text)** | CFG scale variation enables effective branching **without any noise** | **Novel contribution**: deterministic ODE scaling |
| **Coarse Simulation (C.5)** | Using coarser timesteps for reward evaluation causes minimal performance loss | Enables **2-5x reduction** in forward simulation cost |
| **High-Resolution Timesteps (C.5)** | Method scales effectively at 100 timesteps with coarse simulation | Validates method at higher quality settings |

**Critically, these efficiency improvements are complementary.** Combining tighter branching schedules (C.1) with coarse reward simulation (C.5) could enable dramatically more efficient search scaling. We commit to running our full experimental suite with an updated method incorporating these improvements, including on the protein generation task, for the final manuscript.

### Upcoming Experiments

For the final version, we commit to:
- **FLUX image generation experiments** implementing our method on a large-scale flow model with direct comparison to the VP-interpolant approach of Kim et al. (2025)
- **Combined efficiency ablation** demonstrating the multiplicative gains from using tighter schedules + coarse simulation together
- **Extended protein experiments** with efficiency improvements applied

We believe these additions substantially strengthen our contributions and address all core concerns raised during review. We discuss these points in detail in the individual responses below.

---

# Response to Reviewer WCen

We'd like to thank the reviewer for their time and effort in reviewing our work. We are happy to see the recognition that the problem of inference-time optimization is "timely and relevant" and that applying these methods to "novel applications" beyond standard image generation has value. We appreciate the detailed technical assessment and address your specific concerns below.

> The core idea of improving diversity by projecting or orthogonalizing initial Gaussian noise samples is not well justified. In high-dimensional latent spaces, random Gaussian vectors are already nearly orthogonal...

We refer the reviewer to our updated manuscript (Section 4.1), where we have clarified the motivation behind score-orthogonal noise. While particle repulsion and time-decaying noise schedules are indeed used to maximize diversity, **that is not the primary purpose of score-orthogonal noise.**

Fundamentally, flow matching models are trained to satisfy the continuity equation. Sampling with added noise violates this equation due to the divergence introduced by non-orthogonalized noise. The motivation for score-orthogonalized noise is to minimize violations to the theoretical basis of flow matching. While we cannot completely remove divergence in the deterministic case, we have extended the proof in the updated manuscript for the stochastic case using the **Fokker-Planck equation**, providing the rigorous theoretical grounding requested.

> The authors claim that Figure 2 demonstrates a meaningful effect of the method; however, it's unclear if the plotted metric or visual quality genuinely improves with the noise search.

We clarify that Figure 2 illustrates the tradeoff between sample diversity and image quality using different noise schedules; it does not depict the noise search algorithm. Our experiments in **Figures 4, 5, and 6** demonstrate substantial gains in diverse reward functions as compute is scaled with our method, consistently outperforming baselines such as random search.

> The paper overlooks or downplays the findings of Kim et al. (2025)... The submission under review continues to use a linear interpolation in the diffusion process and does not experiment with the VP approach.

We address this in Section 2.1. Flow matching is distinct from diffusion; it utilizes a **linear interpolant** during training and inference to bridge complex distributions. While Kim et al. (2025) convert the flow model to a Variance Preserving (VP) interpolant, this effectively treats the model as a diffusion model. As our goal is to enable scaling specifically for **Flow Matching**, we adhere to the standard linear paradigm. This retains the benefits of flow matching, such as straighter trajectories and fewer required sampling steps. We have updated the manuscript to clearly articulate this distinction and its importance as a key element of our novelty.

**Note on FLUX Experiments:** Due to time constraints during the rebuttal period, we were unable to complete the FLUX experiments comparing against the VP-interpolant approach. We commit to including these experiments in the final version of the manuscript.

> The submission does not explore any adaptive allocation of its NFE (number of function evaluations) budget across the diffusion timeline (Rollover Budget Forcing).

While we do not utilize Rollover Budget Forcing, our **noise search algorithm** fundamentally allocates NFEs based on optimal intermediate samples. In addition, we have included a new ablation in Appendix C.2 (Figure 8) to address your concerns, where we experiment with non-uniform branching schedules (though equal overall compute budgets), using more compute at later timesteps (higher number of branches) in exchange for less branching at earlier timesteps. However we find the standard uniform schedule (ours) tends to perform the best. We agree that the use of RBF (or any adaptive compute allocation strategy) is an interesting direction for future work, and have updated the conclusion accordingly. Note our approach is complementary to RBF and could be updated to incorporate it in future work.

> It is important to compare against methods that directly optimize this metric [Inception Score]... like Ψ-Sampler (Yoon et al., 2025).

While differentiable reward functions allow for efficient steering, our setting does not assume differentiability. The Ψ-Sampler (Yoon et al., 2025) requires a differentiable reward. Comparing the two would be inaccurate, as our method solves a more general "black-box" reward problem. We have revised the manuscript (Section 3.1) to differentiate our problem setup from that of Yoon et al. (2025).

> Questions: Why did you set the timestamp to [0.0, 0.2, 0.4, 0.6, 0.75, 0.8, 0.85, 0.9, 0.95]?

We selected this branching schedule as it increases the compute factor of each branch by approximately 3×, offering efficient scaling. To address the question regarding sensitivity, we have added a comprehensive **Branching Schedule Ablation in Appendix C.1** testing progressively tighter schedules: ~2.15 trajectories `[0.0, 0.4, 0.75, 0.85, 0.9, 0.95]`, ~1.85 trajectories `[0.0, 0.5, 0.8, 0.9, 0.95]`, and ~1.55 trajectories `[0.0, 0.6, 0.9, 0.95]`. Results demonstrate the method remains stable across different schedule choices, **with even the most efficient schedule (~1.55 traj) clearly outperforming random search**.

> Questions: Aside from applying the existing SMC diffusion framework to new tasks, what do you consider the key technical innovation of this work?

We direct the reviewer to our detailed novelty statement in the global response above. In brief: (1) We are the first to apply inference-time scaling to FM while preserving the linear interpolant; (2) We introduce CFG-based branching (Section 5.2), a novel method that enables scaling with the exact deterministic ODE; (3) We provide rigorous theoretical grounding via the Fokker-Planck equation; (4) We demonstrate the first application to scientific domains (protein design); (5) Our ablations reveal efficiency improvements (tighter schedules, coarse simulation) that substantially reduce compute requirements.

**Conclusion:**
We'd like to once again thank the reviewer for their detailed assessment. We believe our new Fokker-Planck proof, the clarification regarding the linear interpolant, the novel CFG-based branching method, and the comprehensive ablation studies address all concerns raised. These additions substantially strengthen the novelty of our work. If you find these improvements satisfactory, we politely ask that the reviewer consider raising their score. We are happy to address any additional questions that may arise.

---

# Response to Reviewer kwmg

We'd like to thank the reviewer for their time and effort in reviewing our work. We are encouraged by the positive assessment and grateful that you found our setting "well-motivated" and recognized that the randomized ODE approach "appears new relative to EDM-style SDE noise." We appreciate that you highlighted the empirical improvements on both ImageNet and protein structures. Below, we address your specific concerns.

> I have a minor concern about the novelty of the paper. The particle guidance repulsion [1] and budget forcing [2] is exactly from respectively previous work.

We acknowledge these influences and do not claim novelty on particle guidance itself. Rather, we utilize a combination of particle guidance and a time-decaying noise schedule to empirically push the diversity-quality Pareto frontier. Our primary novelty lies in **inference-time scaling specifically for flow matching (linear interpolant)**, designing inference algorithms that adhere to this paradigm.

Critically, our revised manuscript includes **a fundamentally new contribution**: CFG-based branching (Section 5.2), which performs inference scaling **without any stochastic noise**, using only variations in the Classifier-Free Guidance scale to create diverse trajectories. This demonstrates that our Noise Search framework is flexible and can leverage entirely different sources of trajectory diversity—a novel finding with practical implications for any conditional flow model using CFG.

> The compute accounting is not aligned across methods, which weakens empirical support.

This is a valid point! While we have not had the time to update the figures to improve on this during the rebuttal, we will update this in the final manuscript. Furthermore, we have added ablations which **substantially reduce our method's compute requirements**:
- **Branching Schedule Ablation (C.1)**: Demonstrates we can use schedules with ~60% less overhead while still outperforming random search
- **Coarse Simulation Ablation (C.5)**: Shows we can use 2-5x coarser timesteps for reward evaluation with minimal performance loss

These efficiency improvements widen our performance margin over random search at matched compute and will be incorporated into the final experiments.

> Ablations on the noise injection are limited. The EDM and SDE ablations are useful but do not disentangle the roles of score orthogonality and particle coupling.

We direct the reviewer to **Figure 2**, which includes a separate **Score-Orthogonal ODE** entry. This entry excludes particle coupling and time-decaying noise. The results suggest score orthogonality alone does not significantly shift the diversity-quality frontier. This is expected, as score-orthogonal noise is intended to minimize divergence of the drift component of the Fokker-Planck equation, not to inject diversity.

> For the orthogonal score, can you quantify the changes in ∇w_t or bound it under a local smoothness model for the learned score?

This is an excellent observation. We have addressed this by replacing the previous proof with a derivation based on the **Fokker-Planck equation** (the stochastic setting of the continuity equation) in **Section 4.1**. This provides a solid theoretical basis that controls for these terms correctly, as detailed in Theorem 4.1 and its proof.

> How sensitive is Random search + Noise Search to the 9 rounds schedule?

The schedule acts as a tunable parameter for trading off compute against search refinement. To demonstrate sensitivity, we have included two comprehensive ablations:
1. **Branching Schedule Ablation (Appendix C.1):** Tests schedules with 4, 5, 6, and 9 rounds, corresponding to ~1.55 to ~3.55 trajectory equivalents. We find only minor loss in scaling performance as schedules become more efficient, still clearly outperforming random search.
2. **Non-Uniform Branching Ablation (Appendix C.2):** Tests allocating more branches to later timesteps where simulation cost is lower. We find uniform branching performs best.

Combined with coarse reward estimation (C.5), these findings suggest we can **dramatically improve computational efficiency** while maintaining performance gains over random search.

> Could you add a matched compute comparison against the VP SDE approach of Kim et al. 2025 [1], both in image and in protein generation?

Due to time constraints during the rebuttal period, we were unable to complete the FLUX experiments comparing against the VP-interpolant approach of Kim et al. (2025). We commit to including experiments on the FLUX image domain, with the approach of Kim et al. (2025) as a baseline, in the finalized manuscript.

**Conclusion:**
We'd like to once again thank the reviewer for the insightful technical questions, particularly regarding the theoretical proof and sensitivity analysis. We believe the updated Fokker-Planck derivation, comprehensive ablations (demonstrating both robustness and efficiency improvements), and the novel CFG-based branching method substantially strengthen our contributions. If the reviewer is satisfied with this response and the updates to the manuscript, we'd ask that they consider raising their score. Thank you!

---

# Response to Reviewer gTSm

We'd like to thank the reviewer for their time and effort in reviewing our work. We are pleased to see the recognition that the proposed algorithms are "conceptually straightforward yet effective" and that our method addresses the important problem of scaling without converting to diffusion-like samplers. Below, we address your specific concerns.

> The paper's main distinction is well-motivated but represents an incremental methodological refinement.

We respectfully argue that establishing inference scaling for **Flow Matching (FM)** is a distinct contribution from diffusion scaling. FM utilizes a linear interpolant, offering different theoretical properties and efficiency benefits (e.g., straighter trajectories) and tackles a more general distribution-to-distribution problem (where the prior distribution x₀ is not necessarily Gaussian). Our work is the first to rigorously explore scaling within this specific paradigm.

Furthermore, we have substantially increased the novelty of our work through this review process. Most notably, we introduce **CFG-based branching (Section 5.2)**, a novel method that enables inference-time scaling using only variations in the Classifier-Free Guidance scale—**without any stochastic noise injection**. This allows scaling while retaining the exact deterministic ODE, a fundamentally different approach from prior work. Combined with our comprehensive ablations demonstrating efficiency improvements and the new Fokker-Planck theoretical grounding, we believe our contributions go well beyond incremental refinement.

> The justification for the DMFM-ODE variant is relatively weak... the claim that linear interpolant scaling outperforms VP-trajectory in flow matching lacks sufficient theoretical or experimental support.

The DMFM-ODE noise schedule was selected to maximize sampling diversity while maintaining sample quality comparable to the deterministic ODE sampler (as shown in Figure 2). We emphasize that our findings show **Noise Search coupled with standard SDE (Euler-Maruyama) noise performs competitively** with the DMFM variant. This is itself a finding of our paper, as it implies an increase in sampling diversity is not necessarily correlated with increasing search scaling effectiveness.

Regarding the comparison to VP-trajectory: we commit to including FLUX experiments with Kim et al. (2025) as a baseline in the final manuscript.

> This paper does not provide any direct head-to-head comparison to the concurrent methods despite heavily citing them.

We acknowledge the lack of linear interpolant baselines, as the field is nascent. Due to time constraints during the rebuttal period, we were unable to complete the FLUX experiments with VP-interpolant comparison. However, we commit to including this comparison in the final manuscript.

> Questions: In image generation, classifier guidance is commonly used to enhance sample fidelity. Did you employ classifier-free guidance (CFG) or any similar technique in your experiments?

We have addressed this concern comprehensively with **two new experiments**:

1. **CFG Compatibility Ablation (Appendix C.3, Figure 10):** We validate our method with CFG scale=1.5, demonstrating full compatibility. Note that CFG sampling improves DINO classification accuracy, causing results to quickly saturate near 100%—this artificial plateau reflects CFG's effectiveness, not a limitation of our method.

2. **CFG-Based Branching (Section 5.2, Main Text, NEW):** We demonstrate that CFG itself can be used as a branching mechanism, where each branch uses a different CFG scale at each branching step. **This allows our method to perform search scaling without stochastic noise, relying entirely on the deterministic ODE while still enabling diversity.** This is a novel approach applicable to any flow model that utilizes CFG. The revised manuscript highlights this significant new finding.

**Conclusion:**
We'd like to once again thank the reviewer for the constructive feedback, which pushed us to strengthen our work. We believe the novel CFG-based branching method (enabling deterministic ODE scaling), the VP-interpolant comparison commitment, the CFG compatibility ablation, and our comprehensive efficiency ablations substantially address your concerns about incrementality and experimental completeness. If these updates are satisfactory, we would appreciate it if you would consider raising your score. We are happy to address any additional questions.

---

# Response to Reviewer AfW2

We'd like to thank the reviewer for their thoughtful and detailed review. We are encouraged that you recognized the clear motivation and appreciated that our method preserves the linear interpolant, retaining "FM's sampling efficiency." We also appreciate the recognition of our results in the scientific domain of protein design—to our knowledge, the first application of inference-time scaling to this area. Below, we address your specific concerns.

> A major concern is the claim of score-orthogonal perturbations. As stated, the notion of "orthogonality to the score" is underspecified statistically.

We completely understand this concern, and we thank you for pushing us to strengthen the theoretical foundation. We have revised the theoretical justification in **Section 4.1** to use the **Fokker-Planck equation**. This provides the correct stochastic alternative to the flow matching continuity equation, resolving the statistical underspecification of the previous proof. The new Theorem 4.1 rigorously shows that score-orthogonal perturbations are divergence-free with respect to the drift term in the Fokker-Planck equation.

> The qualitative results are also unconvincing. Several examples (Figures 14–17) appear weak or inconsistent... The model generates underfitted samples at 1x compute cost.

The image models used are pretrained (Ma et al., 2024) and are capable of generating high quality images (FID-50K of 2.06). The perceived underfitting arises because Ma et al. utilize **250 sampling steps**, while our experiments utilized **20 steps** due to the multiplicative cost of inference scaling.

To address this concern, we have added two ablations:
1. **Increased Timesteps Ablation (Appendix C.5, Figure 12):** Demonstrates similar scaling performance at **100 sampling steps** with 128 samples, verifying our method's effectiveness at higher NFEs.
2. **Coarse Simulation Ablation (Appendix C.5, Figures 11-12):** Shows that when simulating forward to t=1 to obtain rewards, we can approximate the true reward using larger sampling steps for more efficient forward simulation. These ablations demonstrate only minor reductions in scaling performance as the forward simulate compute budget is reduced.

> Can the authors test their method on a large scale generative models (FLUX: image generation, WAN: video generation) with a more practical reward function?

Due to time constraints during the rebuttal period, we were unable to complete the FLUX image generation experiments. We commit to including these experiments in the final version of the manuscript, which will enable testing on a large-scale generative model and direct comparison with the VP-interpolant method of Kim et al. (2025).

> Questions: Injecting stochasticity often necessitates using a smaller discretization step to maintain sample quality. Has the proposed method observed a similar effect?

In **Figure 2 (Section 4.2)**, we include a baseline ODE sampler reference. We observe that for SDE and DMFM-ODE samplers, quality (measured by FID) is retained at 1x compute up to a diversity of approx. ~0.5. This indicates that at the discretization steps we used, the stochasticity does not degrade quality compared to the deterministic baseline. Furthermore, our **Increased Timesteps Ablation (Appendix C.5)** with dt=0.01 (100 timesteps) confirms that the method scales effectively even with finer discretization.

**Conclusion:**
We'd like to once again thank the reviewer for the helpful feedback, which pushed us to solidify our theoretical proofs and add extensive ablations. The new Fokker-Planck derivation provides rigorous theoretical grounding, and our comprehensive ablation studies (Appendix C) address concerns about hyperparameter sensitivity, computational efficiency, and scaling at higher quality settings. While we were unable to complete the FLUX experiments during the rebuttal period, we commit to including them in the final version. We believe these substantial revisions address all concerns raised, and we hope the reviewer will consider raising their score. We are happy to address any additional questions.
