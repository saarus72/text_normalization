# Text normalization

The fully working model is [on huggingface](https://huggingface.co/saarus72/russian_text_normalizer), and a [nice chat](https://huggingface.co/spaces/saarus72/russian-text-normalization) is on HF Space as well.

### Why

A pet project as the (only) [other solution](https://github.com/snakers4/russian_stt_text_normalization) does not seem to be maintainable by either owner or others. Some others do exist and are, like, *bad*.

> E.g. for input `я купил iphone 10X за 14990 руб без 3-x часов полдень и т.д.` output of
> * [russian_stt_text_normalization](https://github.com/snakers4/russian_stt_text_normalization) itself is `я купил ифон десять кс за четыре девять девять ноль рублей без третьи часов полдень и т.д.`
> * [text-normalization-ru-terrible](https://huggingface.co/maximxls/text-normalization-ru-terrible) is `я купил айфон сто икс за тысячу четыреста девяносто рубле без третьих часов пол`,
> * [text-normalization-ru-new](https://huggingface.co/alexue4/text-normalization-ru-new) is `я купил ифон десять икс за четырнадцать тысяч девять`.

### The plan

I went along with! Took these steps:

1. Get a dataset.
    > Done with notebooks to [find](./work/dataset/1_find_numbers.ipynb) and to [itn](./work/dataset/2_inverse_normalize.ipynb) texts, then to [construct dataset](./work/dataset/3_process_itn.ipynb).
    1. Download any vast (informal?) russian raw text corpus. Could be
        * [IlyaGusev/ficbook](https://huggingface.co/datasets/IlyaGusev/ficbook),
        * [IlyaGusev/librusec_full](https://huggingface.co/datasets/IlyaGusev/librusec_full), or
        * ~~[Taiga Corpus](https://tatianashavrina.github.io/taiga_site)~~ [pikabu](https://huggingface.co/datasets/IlyaGusev/pikabu)!
    1. Find occurances w/ regexp patterns like `r"двадцат\S+"`, 
    1. Make sure there is nothing but cyrillic.
    1. Make inverse text normalization (that task is more straightforward and many good solutions do exist).
        * Used ~~[NeMo Text Processing](https://github.com/NVIDIA/NeMo-text-processing)~~ [another python package](https://github.com/flockentanz/word_to_number_ru) with some additions.
    1. Polish things roughly like balance (as `два` seems to be *far* more common than `двумястами`), get rid of ITN mistakes etc.
1. Train an MVP.
    > Done with notebooks to [train](./work/train/train.ipynb) and to [distributed train](./work/train/train-distributed.ipynb) a model.
    1. Get a relatively big LLM as we are going to prune it then (and to onnx it as well so that the resulting performance is compatible with the solution I've mantioned).
        * Seems to be [ai-forever/FRED-T5-1.7B](https://huggingface.co/ai-forever/FRED-T5-1.7B) as it is encoder-decoder, trainable on single **RTX3060 12GB** and good enough to get an MVP.
            > Turned out that 12GB is enough to inference it only so I've trained [ai-forever/FRED-T5-large](https://huggingface.co/ai-forever/FRED-T5-large).

            > I've managed to run **FRED-T5-1.7B** train on two 12GB GPUs using [`tensor_parallel`](https://github.com/BlackSamorez/tensor_parallel) package but model did not perform notably better. Also, the point is to have a small and fast model to infer it on CPU.
    1. Train, like, any barely working model.
        * Several attempts are required as it is not clear which prompt is better. May be
            ```
            <SC1>Было у отца [3]<extra_id_0> сына и [2-3]<extra_id_1> пиджака с блёстками.
            ```
            > Turned out the pattern below works well so I've made no experiments here.
    1. Test and analyze.
    1. ~~Regret deeply.~~
1. To obtain a dataset of a better quality, we want to ask really big smart ass LLM to **(not inverse!)** normalize texts during the training.
    * Unfortunately, LLM exeriments failed. I took instruct models ([Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [ruGPT-3.5 13B LoRA](https://huggingface.co/evilfreelancer/ruGPT-3.5-13B-lora), [GigaSaiga](https://huggingface.co/IlyaGusev/gigasaiga_lora), [Saiga2 7B](https://huggingface.co/IlyaGusev/saiga2_13b_lora)) and plain generation ones ([ruGPT-3.5-13B-GPTQ](https://huggingface.co/fffrrt/ruGPT-3.5-13B-GPTQ) and [Vikhr-7b-0.1](https://huggingface.co/AlexWortega/Vikhr-7b-0.1)), but there were always too much of mistakes which can not be catch automatically. Well, they _were_, so I decided to...
1. Take the [Kaggle Text Normalization Challenge](https://www.kaggle.com/competitions/text-normalization-challenge-russian-language) dataset! So I had latin normalization as well.
    > Done with a notebook to [process kaggle data](./work/dataset/4_process_kaggle.ipynb).
1. Train everything again at last. Put [on hf](https://huggingface.co/saarus72/russian_text_normalizer).

## Inverse Text Normalization

There are but a few packages from namely 
* NVidia's [NeMo](https://github.com/NVIDIA/NeMo-text-processing),
* [Oknolaz](https://github.com/Oknolaz/Russian_w2n),
* [SergeyShk](https://github.com/SergeyShk/Word-to-Number-Russian) and its forks from
    * [averkij](https://github.com/averkij/Word-to-Number-Russian) and
    * [flockentanz](https://github.com/flockentanz/word_to_number_ru).

**NeMo** works well but tends to miss many cases I won't have missed (see the comparison table below). I used it as the first attempt but did my research then.

**Oknolaz** needs to be fed with extracted numbers only and does many mistakes in that case even so bad choice for us.

**SergeyShk** does either
* `replace_groups` — `тысяча сто` to `1100` but `сто двести триста` to `400` or
* `replace` — `сто двести триста` to `100 200 300` but `тысяча сто` to `1000 100`.

It is obvious that addition should be done on decreasing values only so there are some forks to fix it (the overall code is a mess so that I didn't want to do it myself anyway).

**averkij** and **flockentanz** work fine both but have some bugs so I took the second one and fixed them. Also I cover cases like `с половиной` and `одна целая две десятых`.

| Original | 🟡 NeMo TP | 🔴 Oknolaz `replace` | 🔴 SergeyShk `replace_groups` | 🔴 SergeyShk `replace` | 🔴 averkij `replace` | 🔴 flockentanz `replace_groups_sa` | 🟢 flockentanz fixed |
|--|--|--|--|--|--|--|--|
| `сто двести триста да хоть тысячу раз` | 🟢`100 200 300 да хоть 1000 раз` | 🔴`600000` | 🔴`400 да хоть 1000 раз` | 🟢`100 200 300 да хоть 1000 раз` | 🔴`10200 300 да хоть 1000 раз` | 🟢`100 200 300 да хоть 1000 раз` | 🟢`100 200 300 да хоть 1000 раз` |
| `тысяча сто` | 🟢`1100` | 🟢`1100` | 🟢`1100` | 🔴`1000 100` | 🟢`1100` | 🟢`1100` | 🟢`1100` |
| `я видел сто-двести штук` | 🟡`я видел сто-двести штук` | 🔴`300` | 🟢`я видел 100-200 штук` | 🟢`я видел 100-200 штук` | 🟢`я видел 100-200 штук` | 🟢`я видел 100-200 штук` | 🟢`я видел 100-200 штук` |
| `восемь девятьсот двадцать два пять пять пять тридцать пять тридцать пять, лучше позвонить, чем занимать` | 🟡`восемь 922 пять пять пять 35 35 , лучше позвонить, чем занимать` | 🔴`8` | 🔴`115, лучше позвонить, чем занимать` | 🔴`8 900 20 2 5 5 5 30 5 30 5, лучше позвонить, чем занимать` | 🟢`8 922 5 5 5 35 35, лучше позвонить, чем занимать` | 🟢`8 922 5 5 5 35 35, лучше позвонить, чем занимать` | 🟢`8 922 5 5 5 35 35, лучше позвонить, чем занимать` |
| `три с половиной человека` | 🟡`три с половиной человека` | 🔴`3` | 🟡`3 с половиной человека` | 🟡`3 с половиной человека` | 🟢`3.5 человека` | 🟡`3 с половиной человека` | 🟢`3.5 человека` |
| `миллион сто тысяч сто зайцев` | 🟢`1100100 зайцев` | ❌`list index out of range` | 🔴`1000100100 зайцев` | 🔴`1000000 100000 100 зайцев` | `1100100 зайцев` | 🔴`1000100100 зайцев` | 🟢`1100100 зайцев` |
| `одни двойки и ни одной пятёрки` | 🟡`одни двойки и ни одной пятёрки` | 🟡`No valid number words found! ...` | 🟡`1 двойки и ни 1 пятёрки` | 🟡`1 двойки и ни 1 пятёрки` | 🟡`1 двойки и ни 1 пятёрки` | 🟡`1 двойки и ни 1 пятёрки` | 🟡`1 двойки и ни 1 пятёрки` |
| `без одной минуты два` |🟢 `01:59` | 🔴`2` | 🟢`без 1 минуты 2` | 🟢`без 1 минуты 2` | 🟢`без 1 минуты 2` | 🟢`без 1 минуты 2` | 🟢`без 1 минуты 2` |
| `вторая дача пять соток` | 🟡`вторая дача пять соток` | 🔴`5` | 🟢`2 дача 5 соток` | 🟢`2 дача 5 соток` | 🟢`2 дача 5 соток` | 🟢`2 дача 5 соток` | 🟢`2 дача 5 соток` |
| `двести пятьдесят с половиной тысяч отборных солдат Ирака` | 🟡`250 с половиной 1000 отборных солдат Ирака` | 🔴`250000` | 🟡`250 с половиной 1000 отборных солдат Ирака` | 🔴`200 50 с половиной 1000 отборных солдат Ирака` | 🔴`2050000.5 отборных солдат Ирака` | 🟡`250 с половиной 1000 отборных солдат Ирака` | 🟢`250500 отборных солдат Ирака` |
| `ноль целых ноль десятых минус две целых шесть сотых` | 🟢`0,0 -2,06` | 🟡`Redundant number word! ...` | 🔴`0 целых 0.0 минус 2 целых 0.06` | 🔴`0 целых 0.0 минус 2 целых 0.06` | 🔴`0 целых 0.0 минус 2 целых 0.06` | 🔴`0 целых 0.0 минус 2 целых 0.06` | 🟢`0 минус 2.06` |