# LLM Benchmark

Python 3.12 tabanlı, çoktan seçmeli LLM karşılaştırmaları için küçük ve tekrar
üretilebilir bir benchmark çekirdeği. V1 yalnızca deterministik `MockProvider`
kullanır; gerçek veya ücretli API çağrısı yapmaz.

## Kurulum

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,huggingface]"
```

Hugging Face desteği gerekmiyorsa `python -m pip install -e ".[dev]"` yeterlidir.

## Çalıştırma

Yerel fixture ile tamamen çevrimdışı örnek:

```powershell
llm-benchmark run --config configs/mock_smoke.yaml
# veya: python -m llm_benchmark run --config configs/mock_smoke.yaml
```

MMLU-Pro smoke (ilk indirmede ağ, sonrasında Hugging Face cache gerekir):

```powershell
llm-benchmark run --config configs/mmlu_pro_smoke.yaml
```

CLI override örneği:

```powershell
llm-benchmark run --config configs/mock_smoke.yaml --set output_dir=outputs/custom --set seed=7
```

Her koşu `outputs/<run_id>/` altında `results.jsonl`, `summary.json`,
`resolved_config.json`, `dataset_manifest.json` ve `environment.json` üretir.

## Testler

```powershell
pytest
```

Normal test paketi ağ kullanmaz. MMLU-Pro ağ testi isteğe bağlıdır:

```powershell
pytest -m network --run-network
```

## MMLU-Pro

Gerçek veri kaynağı [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
olup örnek config, tekrar üretilebilirlik için tam bir commit revision'ına
sabitlenmiştir. Veri repoya
kopyalanmaz; `datasets` kütüphanesinin cache'i kullanılır. Lisans ve citation
bilgileri upstream veri kartından deney öncesinde doğrulanmalıdır. Smoke/POC
sonuçları resmî tam benchmark skoru değildir.

## Sınırlar

V1; retry yürütme, gerçek provider adapter'ları, streaming, fiyat tablosu,
veritabanı ve web arayüzü içermez. Veri modeli retry ve bilinmeyen token
değerlerini kaydetmeye hazırdır. Ayrıntılar [varsayımlar](docs/assumptions.md)
dosyasındadır.
