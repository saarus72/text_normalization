# Single GPU train [.ipynb](./work/train/train.ipynb)

First model has been trained on singe GPU.

I personally have four RTX 3060 12Gb.
[FRED-T5-1.7B](https://huggingface.co/ai-forever/FRED-T5-1.7B) does not fit into a single GPU.
Surprisingly (not), if not restricted to single GPU, [FRED-T5-large](https://huggingface.co/ai-forever/FRED-T5-large) training causes CUDA OOM.
Seems to be some additional, like, occupation of gradient data leaking from all the GPUs to the main one.
So I have done that `import os; os.environ["CUDA_VISIBLE_DEVICES"] = "0"` thing and used **FRED-T5-large**.

According to the [memory calculator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage), I need these pieces of amount of GPU RAM.

| Model         | Train RAM (f32) | Train RAM (f16) | Inference RAM (f32) | Inference RAM (f16) | Inference RAM (int8) | Inference RAM (int4) |  
|:--------------|:---------------:|:---------------:|:-------------------:|:-------------------:|:--------------------:|:--------------------:|
| FRED-T5-1.7B  | 24.78 GB | 12.39 GB | 6.2 GB | 3.1 GB | 1.55 GB | 792.98 MB |
| FRED-T5-large | 11.46 GB | 5.73 GB | 2.86 GB | 1.43 GB | 733.3 MB | 366.65 MB |
| [ruT5-large](https://huggingface.co/ai-forever/ruT5-large) | 10.99 GB | 5.5 GB | 2.75 GB | 1.37 GB | 703.5 MB | 351.75 MB |
| [ruT5-base](https://huggingface.co/ai-forever/ruT5-base) | 3.32 GB | 1.66 GB | 850.31 MB | 425.15 MB | 212.58 MB | 106.29 MB |

# Two-GPU distributed train [.ipynb](./work/train/train.ipynb)

**transformers** itself suggests [three](https://huggingface.co/docs/transformers/perf_train_gpu_many) options to train on several GPU as a model doesn't fin into a single one.
I chose TensorParallel as I have found a good (but a bit obsolete) package for that :)

The second model has been trained on two GPUs with [`tensor_parallel`](https://github.com/BlackSamorez/tensor_parallel) package.
I suppose I could use 3+ GPUs but there was `Bus error (core dumped)` error (not caused by the library as vanilla train of a _small_ model do cause it as well).
