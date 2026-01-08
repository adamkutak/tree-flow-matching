Dino scoring:
{
  "branch_pairs": {
    "2_1": {
      "fid_score": 46.067142486572266,
      "inception_score": 48.1601676940918,
      "inception_std": 2.4170968532562256,
      "avg_mahalanobis": -1.974422080325894,
      "dino_top1_accuracy": 88.96484375,
      "dino_top5_accuracy": 98.73046875
    },
    "4_1": {
      "fid_score": 43.394287109375,
      "inception_score": 56.33837890625,
      "inception_std": 2.2332041263580322,
      "avg_mahalanobis": -1.9929190793191083,
      "dino_top1_accuracy": 95.21484375,
      "dino_top5_accuracy": 99.8046875
    },
    "8_1": {
      "fid_score": 45.18169021606445,
      "inception_score": 61.53477096557617,
      "inception_std": 3.78969669342041,
      "avg_mahalanobis": -1.95795724139316,
      "dino_top1_accuracy": 98.14453125,
      "dino_top5_accuracy": 100.0
    }
  },
  "scoring_function": "dino_score",
  "sample_method": "noise_search_ode_divfree_max",
  "n_samples": 1024,
  "branch_dt": 0.05,
  "branch_start_time": 0.0,
  "config": {
    "dataset": "imagenet256",
    "device": "cuda",
    "eval_mode": "single_samples",
    "n_samples": 1024,
    "real_samples": 50000,
    "branch_pairs": [
      [
        2,
        1
      ],
      [
        4,
        1
      ],
      [
        8,
        1
      ]
    ],
    "branch_dt": 0.05,
    "branch_start_time": 0.0,
    "sample_method": "noise_search_ode_divfree_max",
    "scoring_function": "dino_score",
    "refinement_batch_size": 32,
    "num_iterations": 1,
    "output_dir": "./results_sweep_2025-08-29_06-07-00",
    "dt_std": 0.7,
    "warp_scale": 0.5,
    "noise_scale": 0.14,
    "lambda_div": 0.9,
    "rounds": 9
  },
  "timestamp": "2025-08-29_14-29-05"
}{
  "branch_pairs": {
    "2_1": {
      "fid_score": 46.43056106567383,
      "inception_score": 48.7623405456543,
      "inception_std": 2.3534963130950928,
      "avg_mahalanobis": -1.9978600362082943,
      "dino_top1_accuracy": 90.8203125,
      "dino_top5_accuracy": 98.4375
    },
    "4_1": {
      "fid_score": 44.50156784057617,
      "inception_score": 53.335548400878906,
      "inception_std": 2.815924644470215,
      "avg_mahalanobis": -1.9709789010230452,
      "dino_top1_accuracy": 94.62890625,
      "dino_top5_accuracy": 99.51171875
    },
    "8_1": {
      "fid_score": 46.42384719848633,
      "inception_score": 62.27632522583008,
      "inception_std": 2.549262762069702,
      "avg_mahalanobis": -2.005663529969752,
      "dino_top1_accuracy": 97.65625,
      "dino_top5_accuracy": 100.0
    }
  },
  "scoring_function": "dino_score",
  "sample_method": "noise_search_sde",
  "n_samples": 1024,
  "branch_dt": 0.05,
  "branch_start_time": 0.0,
  "config": {
    "dataset": "imagenet256",
    "device": "cuda",
    "eval_mode": "single_samples",
    "n_samples": 1024,
    "real_samples": 50000,
    "branch_pairs": [
      [
        2,
        1
      ],
      [
        4,
        1
      ],
      [
        8,
        1
      ]
    ],
    "branch_dt": 0.05,
    "branch_start_time": 0.0,
    "sample_method": "noise_search_sde",
    "scoring_function": "dino_score",
    "refinement_batch_size": 32,
    "num_iterations": 1,
    "output_dir": "./results_sweep_2025-08-29_06-07-00",
    "dt_std": 0.7,
    "warp_scale": 0.5,
    "noise_scale": 0.14,
    "lambda_div": 0.9,
    "rounds": 9
  },
  "timestamp": "2025-08-29_11-55-53"
}{
  "branch_pairs": {
    "2_1": {
      "fid_score": 52.98173904418945,
      "inception_score": 43.34710693359375,
      "inception_std": 2.76894211769104,
      "avg_mahalanobis": -1.9252372103510424,
      "dino_top1_accuracy": 76.5625,
      "dino_top5_accuracy": 90.33203125
    },
    "4_1": {
      "fid_score": 46.33927917480469,
      "inception_score": 50.30347442626953,
      "inception_std": 2.2147905826568604,
      "avg_mahalanobis": -1.854343316750601,
      "dino_top1_accuracy": 85.64453125,
      "dino_top5_accuracy": 97.8515625
    },
    "8_1": {
      "fid_score": 45.91306686401367,
      "inception_score": 55.24619674682617,
      "inception_std": 2.809194803237915,
      "avg_mahalanobis": -1.9181552226655185,
      "dino_top1_accuracy": 91.89453125,
      "dino_top5_accuracy": 99.609375
    }
  },
  "scoring_function": "dino_score",
  "sample_method": "random_search",
  "n_samples": 1024,
  "branch_dt": 0.05,
  "branch_start_time": 0.0,
  "config": {
    "dataset": "imagenet256",
    "device": "cuda",
    "eval_mode": "single_samples",
    "n_samples": 1024,
    "real_samples": 50000,
    "branch_pairs": [
      [
        2,
        1
      ],
      [
        4,
        1
      ],
      [
        8,
        1
      ]
    ],
    "branch_dt": 0.05,
    "branch_start_time": 0.0,
    "sample_method": "random_search",
    "scoring_function": "dino_score",
    "refinement_batch_size": 32,
    "num_iterations": 1,
    "output_dir": "./results_sweep_2025-08-29_06-07-00",
    "dt_std": 0.7,
    "warp_scale": 0.5,
    "noise_scale": 0.14,
    "lambda_div": 0.9,
    "rounds": 9
  },
  "timestamp": "2025-08-29_06-40-01"
}{
  "branch_pairs": {
    "2_1": {
      "fid_score": 46.3289794921875,
      "inception_score": 49.781002044677734,
      "inception_std": 3.0474987030029297,
      "avg_mahalanobis": -2.020241567515768,
      "dino_top1_accuracy": 87.890625,
      "dino_top5_accuracy": 98.4375
    },
    "4_1": {
      "fid_score": 43.738677978515625,
      "inception_score": 58.97318649291992,
      "inception_std": 1.405892252922058,
      "avg_mahalanobis": -1.997660781024024,
      "dino_top1_accuracy": 95.01953125,
      "dino_top5_accuracy": 99.8046875
    },
    "8_1": {
      "fid_score": 45.604698181152344,
      "inception_score": 64.17152404785156,
      "inception_std": 2.5309321880340576,
      "avg_mahalanobis": -1.9977129296166822,
      "dino_top1_accuracy": 97.8515625,
      "dino_top5_accuracy": 100.0
    }
  },
  "scoring_function": "dino_score",
  "sample_method": "random_search_then_noise_search_ode_divfree_max",
  "n_samples": 1024,
  "branch_dt": 0.05,
  "branch_start_time": 0.0,
  "config": {
    "dataset": "imagenet256",
    "device": "cuda",
    "eval_mode": "single_samples",
    "n_samples": 1024,
    "real_samples": 50000,
    "branch_pairs": [
      [
        2,
        1
      ],
      [
        4,
        1
      ],
      [
        8,
        1
      ]
    ],
    "branch_dt": 0.05,
    "branch_start_time": 0.0,
    "sample_method": "random_search_then_noise_search_ode_divfree_max",
    "scoring_function": "dino_score",
    "refinement_batch_size": 32,
    "num_iterations": 1,
    "output_dir": "./results_sweep_2025-08-29_06-07-00",
    "dt_std": 0.7,
    "warp_scale": 0.5,
    "noise_scale": 0.14,
    "lambda_div": 0.9,
    "rounds": 9
  },
  "timestamp": "2025-08-29_17-29-40"
}{
  "timestamp": "2025-08-29_06-07-00",
  "results_dir": "./results_sweep_2025-08-29_06-07-00",
  "dataset": "imagenet256",
  "sample_methods": [
    "random_search",
    "noise_search_sde",
    "noise_search_ode_divfree_max",
    "random_search_then_noise_search_ode_divfree_max"
  ],
  "timestep_configs": [
    [
      20,
      0.05,
      0
    ]
  ],
  "sample_sizes": [
    1024
  ],
  "branch_pairs": "2:1,4:1,8:1",
  "scoring_function": "dino_score",
  "dt_std": 0.7,
  "warp_scale": 0.5,
  "noise_scale": 0.14,
  "lambda_div": 0.9,
  "device": "cuda"


Inception scoring:

{
  "branch_pairs": {
    "2_1": {
      "fid_score": 52.418731689453125,
      "inception_score": 71.8423843383789,
      "inception_std": 3.278064489364624,
      "avg_mahalanobis": -1.972378037578892,
      "dino_top1_accuracy": 74.4140625,
      "dino_top5_accuracy": 87.890625
    },
    "4_1": {
      "fid_score": 54.67594909667969,
      "inception_score": 82.38941192626953,
      "inception_std": 1.9982918500900269,
      "avg_mahalanobis": -1.9736650417326018,
      "dino_top1_accuracy": 76.7578125,
      "dino_top5_accuracy": 89.55078125
    },
    "8_1": {
      "fid_score": 58.98080825805664,
      "inception_score": 88.78175354003906,
      "inception_std": 1.4830809831619263,
      "avg_mahalanobis": -1.9541633105254732,
      "dino_top1_accuracy": 83.49609375,
      "dino_top5_accuracy": 92.3828125
    }
  },
  "scoring_function": "inception_score",
  "sample_method": "noise_search_ode_divfree_max",
  "n_samples": 1024,
  "branch_dt": 0.05,
  "branch_start_time": 0.0,
  "config": {
    "dataset": "imagenet256",
    "device": "cuda",
    "eval_mode": "single_samples",
    "n_samples": 1024,
    "real_samples": 50000,
    "branch_pairs": [
      [
        2,
        1
      ],
      [
        4,
        1
      ],
      [
        8,
        1
      ]
    ],
    "branch_dt": 0.05,
    "branch_start_time": 0.0,
    "sample_method": "noise_search_ode_divfree_max",
    "scoring_function": "inception_score",
    "refinement_batch_size": 32,
    "num_iterations": 1,
    "output_dir": "./results_sweep_2025-08-27_19-51-38",
    "dt_std": 0.7,
    "warp_scale": 0.5,
    "noise_scale": 0.14,
    "lambda_div": 0.9,
    "rounds": 9
  },
  "timestamp": "2025-08-27_22-24-05"
}{
  "branch_pairs": {
    "2_1": {
      "fid_score": 53.4427490234375,
      "inception_score": 73.32763671875,
      "inception_std": 3.9435923099517822,
      "avg_mahalanobis": -1.975264299660921,
      "dino_top1_accuracy": 71.97265625,
      "dino_top5_accuracy": 87.40234375
    },
    "4_1": {
      "fid_score": 55.416507720947266,
      "inception_score": 84.2112045288086,
      "inception_std": 3.0871901512145996,
      "avg_mahalanobis": -1.9986324074561708,
      "dino_top1_accuracy": 79.78515625,
      "dino_top5_accuracy": 92.67578125
    },
    "8_1": {
      "fid_score": 59.97134780883789,
      "inception_score": 91.05591583251953,
      "inception_std": 3.0031192302703857,
      "avg_mahalanobis": -1.9845016830950044,
      "dino_top1_accuracy": 84.1796875,
      "dino_top5_accuracy": 93.45703125
    }
  },
  "scoring_function": "inception_score",
  "sample_method": "random_search_then_noise_search_ode_divfree_max",
  "n_samples": 1024,
  "branch_dt": 0.05,
  "branch_start_time": 0.0,
  "config": {
    "dataset": "imagenet256",
    "device": "cuda",
    "eval_mode": "single_samples",
    "n_samples": 1024,
    "real_samples": 50000,
    "branch_pairs": [
      [
        2,
        1
      ],
      [
        4,
        1
      ],
      [
        8,
        1
      ]
    ],
    "branch_dt": 0.05,
    "branch_start_time": 0.0,
    "sample_method": "random_search_then_noise_search_ode_divfree_max",
    "scoring_function": "inception_score",
    "refinement_batch_size": 32,
    "num_iterations": 1,
    "output_dir": "./results_sweep_2025-08-27_19-51-38",
    "dt_std": 0.7,
    "warp_scale": 0.5,
    "noise_scale": 0.14,
    "lambda_div": 0.9,
    "rounds": 9
  },
  "timestamp": "2025-08-28_01-22-32"
}{
  "timestamp": "2025-08-27_19-51-38",
  "results_dir": "./results_sweep_2025-08-27_19-51-38",
  "dataset": "imagenet256",
  "sample_methods": [
    "noise_search_ode_divfree_max",
    "random_search_then_noise_search_ode_divfree_max"
  ],
  "timestep_configs": [
    [
      20,
      0.05,
      0
    ]
  ],
  "sample_sizes": [
    1024
  ],
  "branch_pairs": "2:1,4:1,8:1",
  "scoring_function": "inception_score",
  "dt_std": 0.7,
  "warp_scale": 0.5,
  "noise_scale": 0.14,
  "lambda_div": 0.9,
  "device": "cuda"
}
{
  "branch_pairs": {
    "2_1": {
      "fid_score": 51.34600830078125,
      "inception_score": 72.76887512207031,
      "inception_std": 3.030940055847168,
      "avg_mahalanobis": -1.9821743671782315,
      "dino_top1_accuracy": 74.4140625,
      "dino_top5_accuracy": 90.234375
    },
    "4_1": {
      "fid_score": 51.339725494384766,
      "inception_score": 82.59681701660156,
      "inception_std": 1.4567745923995972,
      "avg_mahalanobis": -1.9479448051424697,
      "dino_top1_accuracy": 83.203125,
      "dino_top5_accuracy": 93.84765625
    },
    "8_1": {
      "fid_score": 55.88508224487305,
      "inception_score": 89.6324462890625,
      "inception_std": 2.928668975830078,
      "avg_mahalanobis": -1.9875003143097274,
      "dino_top1_accuracy": 85.15625,
      "dino_top5_accuracy": 94.3359375
    }
  },
  "scoring_function": "inception_score",
  "sample_method": "noise_search_sde",
  "n_samples": 1024,
  "branch_dt": 0.05,
  "branch_start_time": 0.0,
  "config": {
    "dataset": "imagenet256",
    "device": "cuda",
    "eval_mode": "single_samples",
    "n_samples": 1024,
    "real_samples": 50000,
    "branch_pairs": [
      [
        2,
        1
      ],
      [
        4,
        1
      ],
      [
        8,
        1
      ]
    ],
    "branch_dt": 0.05,
    "branch_start_time": 0.0,
    "sample_method": "noise_search_sde",
    "scoring_function": "inception_score",
    "refinement_batch_size": 32,
    "num_iterations": 1,
    "output_dir": "./results_sweep_2025-08-28_18-21-41",
    "dt_std": 0.7,
    "warp_scale": 0.5,
    "noise_scale": 0.14,
    "lambda_div": 0.9,
    "rounds": 9
  },
  "timestamp": "2025-08-28_23-30-13"
}
