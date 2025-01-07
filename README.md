# 🔥 TrustRAG: Enhancing Robustness and Trustworthiness in RAG
[[Project page]](https://trust-rag.github.io/) [[Paper]](https://arxiv.org/pdf/2501.00879)

[Huichi Zhou](https://huichizhou.github.io/)<sup>1</sup>,[Kin-Hei Lee](https://openreview.net/profile?id=~Lee_KinHei1)<sup>1</sup>,[Zhonghao Zhan](https://zhonghaozhan.github.io/)<sup>1</sup>,[Yue Chen](https://openreview.net/profile?id=~Yue_Chen8)<sup>2</sup>,[Zhenhao Li](https://zhenhaoli.net/)<sup>1</sup>,

<sup>1</sup>Imperial College London,<sup>2</sup>Peking University

<img src="media/Method.jpg" alt="drawing" width="100%"/>

## 🛝 Try it out!

### 🛠️ Installation

```
git clone https://github.com/HuichiZhou/TrustRAG.git

conda create -n trustrag python=3.10

conda activate trustrag

pip install lmdeploy

pip install beir

pip install nltk

pip install rouge_score

python run_trustrag.py
```

## 🙏 Acknowledgement

* Our code used the implementation of [corpus-poisoning](https://github.com/princeton-nlp/corpus-poisoning).
* The model part of our code is from [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection).
* Our code used [beir](https://github.com/beir-cellar/beir) benchmark.
* Our code used [contriever](https://github.com/facebookresearch/contriever) for retrieval augmented generation (RAG).
* Our code used [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) for corpus poisoning attack.


## 📝 Citation and Reference

If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:

```
@article{zhou2025trustrag,
  title={TrustRAG: Enhancing Robustness and Trustworthiness in RAG},
  author={Zhou, Huichi and Lee, Kin-Hei and Zhan, Zhonghao and Chen, Yue and Li, Zhenhao},
  journal={arXiv preprint arXiv:2501.00879},
  year={2025}
}
```

```
@misc{zou2024poisonedrag,
      title={PoisonedRAG: Knowledge Poisoning Attacks to Retrieval-Augmented Generation of Large Language Models}, 
      author={Wei Zou and Runpeng Geng and Binghui Wang and Jinyuan Jia},
      year={2024},
      eprint={2402.07867},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
