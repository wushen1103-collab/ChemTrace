# Phase-0 Manuscript Tables

## Certificate Summary

| artifact                                            |   n_tasks |   n_affected | relations                                 | effect_level                          | retrieval_level                            | best_effect_task   | best_effect_relation   | best_effect_ci            |
|:----------------------------------------------------|----------:|-------------:|:------------------------------------------|:--------------------------------------|:-------------------------------------------|:-------------------|:-----------------------|:--------------------------|
| phase0_gin_bbbp_stereo_changed_aff82_d20_lowtrain05 |         1 |           82 | exact,random,stereo,decoy                 | no_stable_positive_inflation_detected | semantic_lattice_recovers_changed_variants | bbbp               | stereo                 | 0.0028 [-0.0653, 0.0654]  |
| phase0_gru_bbbp_stereo_changed_aff82_d20_lowtrain05 |         1 |           82 | exact,random,stereo,decoy                 | no_stable_positive_inflation_detected | semantic_lattice_recovers_changed_variants | bbbp               | exact                  | 0.0182 [-0.0048, 0.0660]  |
| phase0_tf_aff128_probe                              |         5 |          640 | exact,random,parent,decoy                 | directional_positive_not_ci_confirmed | not_evaluated                              | tox21_mmp          | parent                 | 0.0230 [-0.0124, 0.0570]  |
| phase0_tf_bbbp_stereo_changed_aff82_d20             |         1 |           82 | exact,random,stereo,decoy                 | directional_positive_not_ci_confirmed | semantic_lattice_recovers_changed_variants | bbbp               | exact                  | 0.0500 [-0.0338, 0.1326]  |
| phase0_tf_bbbp_stereo_changed_aff82_d20_lowtrain02  |         1 |           82 | exact,random,stereo,decoy                 | no_stable_positive_inflation_detected | semantic_lattice_recovers_changed_variants | bbbp               | exact                  | -0.0039 [-0.0331, 0.0238] |
| phase0_tf_bbbp_stereo_changed_aff82_d20_lowtrain05  |         1 |           82 | exact,random,stereo,decoy                 | no_stable_positive_inflation_detected | semantic_lattice_recovers_changed_variants | bbbp               | stereo                 | 0.0061 [-0.0270, 0.0386]  |
| phase0_tf_bbbp_stereo_changed_aff82_d20_lowtrain25  |         1 |           82 | exact,random,stereo,decoy                 | no_stable_positive_inflation_detected | semantic_lattice_recovers_changed_variants | bbbp               | stereo                 | 0.0035 [-0.0462, 0.0514]  |
| phase0_tf_confirm_top_len158_extra                  |         9 |          576 | exact,random,parent,stereo,tautomer,decoy | directional_positive_not_ci_confirmed | semantic_lattice_recovers_changed_variants | bace               | random                 | 0.0514 [-0.0349, 0.1683]  |
| phase0_tf_mmp_aff128_isolated                       |         1 |          128 | exact,parent,decoy                        | no_stable_positive_inflation_detected | not_evaluated                              | tox21_mmp          | parent                 | 0.0042 [-0.0317, 0.0378]  |
| phase0_tf_mmp_stereo_changed_aff128_d20             |         1 |          128 | exact,stereo,decoy                        | no_stable_positive_inflation_detected | semantic_lattice_recovers_changed_variants | tox21_mmp          | exact                  | 0.0031 [-0.0261, 0.0305]  |

## Retrieval Lattice Summary

| audit                         | injection_relation   |   dose | detector   |   injection_recall |   changed_only_recall |   clean_hit_rate |
|:------------------------------|:---------------------|-------:|:-----------|-------------------:|----------------------:|-----------------:|
| bbbp_stereo_changed_aff82_d20 | decoy                |     20 | canonical  |         0          |                     0 |        0         |
| bbbp_stereo_changed_aff82_d20 | decoy                |     20 | parent     |         0          |                     0 |        0.280488  |
| bbbp_stereo_changed_aff82_d20 | decoy                |     20 | raw        |         0          |                     0 |        0         |
| bbbp_stereo_changed_aff82_d20 | decoy                |     20 | stereo     |         0          |                     0 |        0.268293  |
| bbbp_stereo_changed_aff82_d20 | decoy                |     20 | tautomer   |         0          |                     0 |        0.0609756 |
| bbbp_stereo_changed_aff82_d20 | exact                |     20 | canonical  |         1          |                       |        0         |
| bbbp_stereo_changed_aff82_d20 | exact                |     20 | raw        |         0.0121951  |                       |        0         |
| bbbp_stereo_changed_aff82_d20 | random               |     20 | canonical  |         1          |                       |        0         |
| bbbp_stereo_changed_aff82_d20 | random               |     20 | raw        |         0          |                       |        0         |
| bbbp_stereo_changed_aff82_d20 | stereo               |     20 | canonical  |         0          |                     0 |        0         |
| bbbp_stereo_changed_aff82_d20 | stereo               |     20 | raw        |         0          |                     0 |        0         |
| bbbp_stereo_changed_aff82_d20 | stereo               |     20 | stereo     |         1          |                     1 |        0.268293  |
| confirm_top                   | decoy                |      1 | canonical  |         0          |                     0 |        0         |
| confirm_top                   | decoy                |      1 | parent     |         0          |                     0 |        0.107639  |
| confirm_top                   | decoy                |      1 | raw        |         0          |                     0 |        0         |
| confirm_top                   | decoy                |      1 | stereo     |         0          |                     0 |        0.0607639 |
| confirm_top                   | decoy                |      1 | tautomer   |         0          |                     0 |        0.0277778 |
| confirm_top                   | decoy                |      5 | canonical  |         0          |                     0 |        0         |
| confirm_top                   | decoy                |      5 | parent     |         0          |                     0 |        0.107639  |
| confirm_top                   | decoy                |      5 | raw        |         0          |                     0 |        0         |
| confirm_top                   | decoy                |      5 | stereo     |         0          |                     0 |        0.0607639 |
| confirm_top                   | decoy                |      5 | tautomer   |         0          |                     0 |        0.0277778 |
| confirm_top                   | decoy                |     20 | canonical  |         0          |                     0 |        0         |
| confirm_top                   | decoy                |     20 | parent     |         0          |                     0 |        0.107639  |
| confirm_top                   | decoy                |     20 | raw        |         0          |                     0 |        0         |
| confirm_top                   | decoy                |     20 | stereo     |         0          |                     0 |        0.0607639 |
| confirm_top                   | decoy                |     20 | tautomer   |         0          |                     0 |        0.0277778 |
| confirm_top                   | exact                |      1 | canonical  |         1          |                       |        0         |
| confirm_top                   | exact                |      1 | raw        |         0.432292   |                       |        0         |
| confirm_top                   | exact                |      5 | canonical  |         1          |                       |        0         |
| confirm_top                   | exact                |      5 | raw        |         0.432292   |                       |        0         |
| confirm_top                   | exact                |     20 | canonical  |         1          |                       |        0         |
| confirm_top                   | exact                |     20 | raw        |         0.432292   |                       |        0         |
| confirm_top                   | parent               |      1 | canonical  |         0.737847   |                     0 |        0         |
| confirm_top                   | parent               |      1 | parent     |         1          |                     1 |        0.107639  |
| confirm_top                   | parent               |      1 | raw        |         0.409722   |                     0 |        0         |
| confirm_top                   | parent               |      5 | canonical  |         0.737847   |                     0 |        0         |
| confirm_top                   | parent               |      5 | parent     |         1          |                     1 |        0.107639  |
| confirm_top                   | parent               |      5 | raw        |         0.409722   |                     0 |        0         |
| confirm_top                   | parent               |     20 | canonical  |         0.737847   |                     0 |        0         |
| confirm_top                   | parent               |     20 | parent     |         1          |                     1 |        0.107639  |
| confirm_top                   | parent               |     20 | raw        |         0.409722   |                     0 |        0         |
| confirm_top                   | random               |      1 | canonical  |         1          |                       |        0         |
| confirm_top                   | random               |      1 | raw        |         0.00347222 |                       |        0         |
| confirm_top                   | random               |      5 | canonical  |         1          |                       |        0         |
| confirm_top                   | random               |      5 | raw        |         0.0104167  |                       |        0         |
| confirm_top                   | random               |     20 | canonical  |         1          |                       |        0         |
| confirm_top                   | random               |     20 | raw        |         0.0451389  |                       |        0         |
| confirm_top                   | stereo               |      1 | canonical  |         0.699653   |                     0 |        0         |
| confirm_top                   | stereo               |      1 | raw        |         0.350694   |                     0 |        0         |
| confirm_top                   | stereo               |      1 | stereo     |         1          |                     1 |        0.0607639 |
| confirm_top                   | stereo               |      5 | canonical  |         0.699653   |                     0 |        0         |
| confirm_top                   | stereo               |      5 | raw        |         0.350694   |                     0 |        0         |
| confirm_top                   | stereo               |      5 | stereo     |         1          |                     1 |        0.0607639 |
| confirm_top                   | stereo               |     20 | canonical  |         0.699653   |                     0 |        0         |
| confirm_top                   | stereo               |     20 | raw        |         0.350694   |                     0 |        0         |
| confirm_top                   | stereo               |     20 | stereo     |         1          |                     1 |        0.0607639 |
| confirm_top                   | tautomer             |      1 | canonical  |         0.8125     |                     0 |        0         |
| confirm_top                   | tautomer             |      1 | raw        |         0.381944   |                     0 |        0         |
| confirm_top                   | tautomer             |      1 | tautomer   |         1          |                     1 |        0.0277778 |
| confirm_top                   | tautomer             |      5 | canonical  |         0.8125     |                     0 |        0         |
| confirm_top                   | tautomer             |      5 | raw        |         0.381944   |                     0 |        0         |
| confirm_top                   | tautomer             |      5 | tautomer   |         1          |                     1 |        0.0277778 |
| confirm_top                   | tautomer             |     20 | canonical  |         0.8125     |                     0 |        0         |
| confirm_top                   | tautomer             |     20 | raw        |         0.381944   |                     0 |        0         |
| confirm_top                   | tautomer             |     20 | tautomer   |         1          |                     1 |        0.0277778 |
| mmp_stereo_changed_aff128_d20 | decoy                |     20 | canonical  |         0          |                     0 |        0         |
| mmp_stereo_changed_aff128_d20 | decoy                |     20 | parent     |         0          |                     0 |        0.1875    |
| mmp_stereo_changed_aff128_d20 | decoy                |     20 | raw        |         0          |                     0 |        0         |
| mmp_stereo_changed_aff128_d20 | decoy                |     20 | stereo     |         0          |                     0 |        0.414062  |
| mmp_stereo_changed_aff128_d20 | decoy                |     20 | tautomer   |         0          |                     0 |        0.078125  |
| mmp_stereo_changed_aff128_d20 | exact                |     20 | canonical  |         1          |                       |        0         |
| mmp_stereo_changed_aff128_d20 | exact                |     20 | raw        |         0.859375   |                       |        0         |
| mmp_stereo_changed_aff128_d20 | stereo               |     20 | canonical  |         0          |                     0 |        0         |
| mmp_stereo_changed_aff128_d20 | stereo               |     20 | raw        |         0          |                     0 |        0         |
| mmp_stereo_changed_aff128_d20 | stereo               |     20 | stereo     |         1          |                     1 |        0.414062  |

## Retrieval Baseline SOTA Summary

| source                                 | relation_group   |   cells |   lattice_changed_recall_mean |   best_exact_changed_recall_mean |   best_similarity_changed_recall_mean |   lattice_gain_vs_exact_mean |   lattice_gain_vs_similarity_mean |   lattice_wins_vs_exact |   lattice_wins_vs_similarity |   lattice_ties_vs_similarity |
|:---------------------------------------|:-----------------|--------:|------------------------------:|---------------------------------:|--------------------------------------:|-----------------------------:|----------------------------------:|------------------------:|-----------------------------:|-----------------------------:|
| confirm_top_sample5k_clean_background  | semantic_all     |      78 |                             1 |                                0 |                              0.841012 |                            1 |                          0.159188 |                      78 |                           39 |                           39 |
| confirm_top_sample5k_clean_background  | parent           |      27 |                             1 |                                0 |                              0.88947  |                            1 |                          0.111109 |                      27 |                           12 |                           15 |
| confirm_top_sample5k_clean_background  | stereo           |      24 |                             1 |                                0 |                              1        |                            1 |                          0        |                      24 |                            0 |                           24 |
| confirm_top_sample5k_clean_background  | tautomer         |      27 |                             1 |                                0 |                              0.651231 |                            1 |                          0.348769 |                      27 |                           27 |                            0 |
| changed_only_sample5k_clean_background | semantic_all     |       2 |                             1 |                                0 |                              1        |                            1 |                          0        |                       2 |                            0 |                            2 |
| changed_only_sample5k_clean_background | stereo           |       2 |                             1 |                                0 |                              1        |                            1 |                          0        |                       2 |                            0 |                            2 |

## Retrieval Baseline Bootstrap

| source               | relation_group   |   cells |   clusters | lattice_gain_vs_exact_ci   | lattice_gain_vs_exact_ci_positive   | lattice_gain_vs_similarity_ci   | lattice_gain_vs_similarity_ci_positive   | lattice_win_rate_vs_similarity_ci   | lattice_tie_rate_vs_similarity_ci   |
|:---------------------|:-----------------|--------:|-----------:|:---------------------------|:------------------------------------|:--------------------------------|:-----------------------------------------|:------------------------------------|:------------------------------------|
| changed_sample5k     | semantic_all     |       2 |          2 | 1.0000 [1.0000, 1.0000]    | True                                | 0.0000 [0.0000, 0.0000]         | False                                    | 0.0000 [0.0000, 0.0000]             | 1.0000 [1.0000, 1.0000]             |
| changed_sample5k     | stereo           |       2 |          2 | 1.0000 [1.0000, 1.0000]    | True                                | 0.0000 [0.0000, 0.0000]         | False                                    | 0.0000 [0.0000, 0.0000]             | 1.0000 [1.0000, 1.0000]             |
| confirm_top_sample5k | parent           |      27 |          9 | 1.0000 [1.0000, 1.0000]    | True                                | 0.1111 [0.0159, 0.2169]         | True                                     | 0.4444 [0.1111, 0.7778]             | 0.5556 [0.2222, 0.8889]             |
| confirm_top_sample5k | semantic_all     |      78 |          9 | 1.0000 [1.0000, 1.0000]    | True                                | 0.1592 [0.1101, 0.2181]         | True                                     | 0.5000 [0.4000, 0.6000]             | 0.5000 [0.4000, 0.6000]             |
| confirm_top_sample5k | stereo           |      24 |          8 | 1.0000 [1.0000, 1.0000]    | True                                | 0.0000 [0.0000, 0.0000]         | False                                    | 0.0000 [0.0000, 0.0000]             | 1.0000 [1.0000, 1.0000]             |
| confirm_top_sample5k | tautomer         |      27 |          9 | 1.0000 [1.0000, 1.0000]    | True                                | 0.3488 [0.2066, 0.5241]         | True                                     | 1.0000 [1.0000, 1.0000]             | 0.0000 [0.0000, 0.0000]             |

## Downstream Effect Robustness

| artifact                                            | task         | relation   |   dose |   n_seeds | label_adjusted_effect_ci   | linear_adjusted_effect_ci   | linear_affected_coef_minus_decoy_support   | ci_positive   |
|:----------------------------------------------------|:-------------|:-----------|-------:|----------:|:---------------------------|:----------------------------|:-------------------------------------------|:--------------|
| phase0_tf_bbbp_stereo_changed_aff82_d20             | bbbp         | exact      |     20 |        12 | 0.0500 [-0.0168, 0.1120]   | 0.0336 [-0.0102, 0.0792]    | directional_positive                       | False         |
| phase0_tf_confirm_top_len158_extra                  | bace         | random     |      1 |         4 | 0.0263 [-0.0550, 0.1103]   | 0.0198 [-0.0442, 0.0803]    | unstable_or_null                           | False         |
| phase0_tf_aff128_probe                              | tox21_mmp    | parent     |      1 |        12 | 0.0230 [-0.0022, 0.0476]   | 0.0188 [0.0009, 0.0370]     | ci_positive                                | True          |
| phase0_gru_bbbp_stereo_changed_aff82_d20_lowtrain05 | bbbp         | exact      |     20 |        12 | 0.0182 [-0.0034, 0.0561]   | 0.0073 [-0.0035, 0.0250]    | unstable_or_null                           | False         |
| phase0_tf_bbbp_stereo_changed_aff82_d20             | bbbp         | random     |     20 |        12 | 0.0157 [-0.0417, 0.0573]   | 0.0081 [-0.0237, 0.0386]    | unstable_or_null                           | False         |
| phase0_tf_aff128_probe                              | tox21_are    | exact      |     20 |         4 | 0.0155 [-0.0143, 0.0452]   | 0.0118 [-0.0012, 0.0248]    | directional_positive                       | False         |
| phase0_tf_aff128_probe                              | bbbp         | random     |     20 |         4 | 0.0149 [-0.0066, 0.0389]   | 0.0134 [-0.0011, 0.0321]    | directional_positive                       | False         |
| phase0_tf_confirm_top_len158_extra                  | bbbp         | random     |     20 |        12 | 0.0144 [-0.0063, 0.0336]   | 0.0148 [0.0001, 0.0290]     | ci_positive                                | True          |
| phase0_tf_confirm_top_len158_extra                  | tox21_mmp    | parent     |      1 |        12 | 0.0142 [-0.0321, 0.0615]   | 0.0221 [-0.0050, 0.0477]    | directional_positive                       | False         |
| phase0_tf_bbbp_stereo_changed_aff82_d20             | bbbp         | stereo     |     20 |        12 | 0.0127 [-0.0375, 0.0616]   | -0.0063 [-0.0401, 0.0270]   | unstable_or_null                           | False         |
| phase0_tf_confirm_top_len158_extra                  | tox21_ahr    | random     |      1 |        12 | 0.0125 [-0.0190, 0.0480]   | 0.0089 [-0.0116, 0.0256]    | directional_positive                       | False         |
| phase0_tf_confirm_top_len158_extra                  | hiv          | exact      |     20 |        12 | 0.0117 [-0.0205, 0.0459]   | 0.0176 [-0.0132, 0.0501]    | directional_positive                       | False         |
| phase0_tf_confirm_top_len158_extra                  | tox21_are    | exact      |     20 |        12 | 0.0098 [-0.0224, 0.0407]   | 0.0072 [-0.0156, 0.0286]    | unstable_or_null                           | False         |
| phase0_tf_confirm_top_len158_extra                  | tox21_ahr    | exact      |     20 |        12 | 0.0091 [-0.0555, 0.0666]   | 0.0113 [-0.0344, 0.0516]    | unstable_or_null                           | False         |
| phase0_tf_aff128_probe                              | hiv          | exact      |     20 |         4 | 0.0081 [-0.0376, 0.0493]   | -0.0100 [-0.0366, 0.0139]   | unstable_or_null                           | False         |
| phase0_tf_aff128_probe                              | tox21_ahr    | exact      |     20 |         4 | 0.0071 [-0.0094, 0.0344]   | 0.0027 [-0.0103, 0.0193]    | unstable_or_null                           | False         |
| phase0_tf_bbbp_stereo_changed_aff82_d20_lowtrain05  | bbbp         | stereo     |     20 |        24 | 0.0061 [-0.0204, 0.0339]   | -0.0035 [-0.0215, 0.0140]   | unstable_or_null                           | False         |
| phase0_tf_bbbp_stereo_changed_aff82_d20_lowtrain05  | bbbp         | exact      |     20 |        24 | 0.0057 [-0.0180, 0.0292]   | -0.0023 [-0.0198, 0.0163]   | unstable_or_null                           | False         |
| phase0_tf_confirm_top_len158_extra                  | sider_gastro | exact      |      1 |        12 | 0.0045 [-0.0097, 0.0184]   | 0.0058 [-0.0065, 0.0173]    | directional_positive                       | False         |
| phase0_tf_mmp_aff128_isolated                       | tox21_mmp    | parent     |      1 |        12 | 0.0042 [-0.0150, 0.0237]   | 0.0019 [-0.0124, 0.0173]    | unstable_or_null                           | False         |
| phase0_tf_bbbp_stereo_changed_aff82_d20_lowtrain05  | bbbp         | random     |     20 |        12 | 0.0038 [-0.0265, 0.0361]   | 0.0059 [-0.0169, 0.0299]    | unstable_or_null                           | False         |
| phase0_tf_bbbp_stereo_changed_aff82_d20_lowtrain25  | bbbp         | stereo     |     20 |        12 | 0.0035 [-0.0288, 0.0361]   | 0.0209 [0.0037, 0.0380]     | ci_positive                                | True          |
| phase0_tf_mmp_stereo_changed_aff128_d20             | tox21_mmp    | exact      |     20 |        12 | 0.0031 [-0.0210, 0.0246]   | 0.0022 [-0.0177, 0.0202]    | unstable_or_null                           | False         |
| phase0_gin_bbbp_stereo_changed_aff82_d20_lowtrain05 | bbbp         | stereo     |     20 |        12 | 0.0028 [-0.0560, 0.0610]   | 0.0125 [-0.0020, 0.0250]    | directional_positive                       | False         |
| phase0_tf_confirm_top_len158_extra                  | clintox_tox  | exact      |      1 |         4 | 0.0021 [-0.0105, 0.0145]   | 0.0010 [-0.0044, 0.0056]    | unstable_or_null                           | False         |

Summary: no decoy-relative downstream effect cell has a bootstrap CI entirely above zero; the strongest result family supports semantic retrieval certificates more strongly than universal performance-inflation claims.
