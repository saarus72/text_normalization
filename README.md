# Text normalization

### Why

A pet project as the (only) [other solution](https://github.com/snakers4/russian_stt_text_normalization) does not seem to be maintainable by either owner or others.

### Do I have a plan

I from where I stand see these steps:

1. Get a dataset.
    1. Download any vast (informal?) russian raw text corpus. Could be
        * [IlyaGusev/ficbook](https://huggingface.co/datasets/IlyaGusev/ficbook),
        * [IlyaGusev/librusec_full](https://huggingface.co/datasets/IlyaGusev/librusec_full), or
        * [Taiga Corpus](https://tatianashavrina.github.io/taiga_site).
    1. Find occurances w/ regexp patterns like `r"двадцат\S+"`, 
    1. Make sure there is nothing but cyrillic.
    1. Make inverse text normalization (that task is more straightforward and many good solutions do exist).
    1. Polish things roughly like balance (as `два` seems to be *far* more common than `двумястами`), get rid of ITN mistakes etc.
1. Train an MVP.
    1. Get a relatively big LLM as we are going to prune it then (and to onnx as well so that the resulting performance is compatible with the solution I've mantioned).
        * Seems to be [ai-forever/FRED-T5-1.7B](https://huggingface.co/ai-forever/FRED-T5-1.7B) as it is encoder-decoder, trainable on single **RTX3060 12Gb** and good enough to get an MVP.
    1. Train, like, any barely working model.
        * Several attempts are required as it is not clear which prompt is better. May be
            ```
            <SC1>Было у отца [3]<extra_id_0> сына и [2-3]<extra_id_1> пиджака с блёстками.
            ```
    1. Test and analyze.
    1. ~~Regret deeply.~~
1. To obtain a dataset of a better quality, we want to ask really big smart ass LLM to **(not inverse!)** normalize texts during the training. It may benefit in
    * having latin staff normalization (abbreviations, brands, urls etc.),
    * cover non-trivial numbers like dates `в период 20-24.10.23 отключат всё и сразу` as we are unlikely to construct such strings on owr own, and
    * balance numbers as we seem to may change **some** digits freely to have a *bigger* number (from `до 3-го предупреждения` to `до 1488-го предупреждения` but not  `получил 3 предупреждения` to `получил 1488 предупреждения`; increase on a multiple of 100 looks nice).
1. Finetune again.
1. GOTO 1 or 3.