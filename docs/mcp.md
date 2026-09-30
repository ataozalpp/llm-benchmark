# MCP calculator bağlantısı

Bu belge mevcut yerel MCP demosunun kurulmasını ve doğrulanmasını anlatır.
Amaç, mevcut calculator iş mantığını ayrı bir FastMCP process'i üzerinden
çağırmak ve cevabı benchmark'ın kendi araç sözleşmesine dönüştürmektir.

## Mevcut kapsam

- Tek yerel FastMCP server ve `calculator` aracı.
- Ayrı Python ortamları arasında stdio iletişimi.
- Legacy initialization yolu ve tools capability kontrolü.
- Tek sayfalı, tek araçlı katalog keşfi ve explicit allowlist.
- Mevcut sınırlı JSON Schema profiliyle input doğrulaması.
- Structured JSON object sonucunun `ToolResult` dönüşümü.
- In-process smoke, gerçek stdio smoke ve negatif argüman testleri.

Bu demo gerçek LLM çağırmaz. `17 * 23 = 391` sonucu yazılım entegrasyonunu
doğrular; model kalitesi ölçümü değildir. MCP henüz `tool-run`, agent loop,
API veya worker çalıştırma yoluna bağlanmamıştır.

## Mimari ve dosyalar

```text
Benchmark ortamı (.venv)
    demo_mcp_stdio_smoke
    → mcp_client: session ve discovery
    → stdio
Server ortamı (.venv-mcp)
    demo_mcp_server: FastMCP
    → CalculatorArguments
    → calculator_handler
    → structured JSON cevap
Benchmark ortamı
    → mcp_mapping
    → ToolResult
```

| Dosya | Sorumluluk |
|---|---|
| `src/llm_benchmark/demo_mcp_server.py` | Calculator'ı FastMCP aracı olarak sunar |
| `src/llm_benchmark/demo_mcp_smoke.py` | Aynı process içinde server davranışını kontrol eder |
| `src/llm_benchmark/mcp_client.py` | Güvenilir yerel launch profili, session ve katalog kabulü |
| `src/llm_benchmark/mcp_mapping.py` | SDK'dan bağımsız descriptor/sonuç dönüşümleri |
| `src/llm_benchmark/demo_mcp_stdio_smoke.py` | Gerçek subprocess ile uçtan uca smoke |
| `tests/test_mcp_mapping.py` | Saf dönüşüm ve çıktı bütçesi kontrolleri |
| `tests/test_mcp_client.py` | Tek sayfalı katalog kabul/ret kontrolleri |
| `tests/test_mcp_stdio.py` | Gerçek bağlantı ve yanlış girdilerin reddi |
| `requirements-mcp-server.txt` | Server framework sürümü |

## Ortamlar neden ayrı?

Mevcut backend FastAPI bağımlılıkları ile seçilen FastMCP 4.0.4 server
bağımlılıklarının Starlette aralıkları uyuşmadığı için iki ortam kullanılır.
Backend'in MCP SDK client'ı `.venv`, FastMCP server `.venv-mcp` içinde çalışır.
İletişim Python nesnelerini paylaşarak değil stdio üzerinden gerçekleşir.
Server requirements dosyası framework sürümünü sabitler; tam transitif lock değildir.

## Kurulum — Windows PowerShell

Komutları depo kökünde çalıştır. Proje Python 3.12 gerektirir. Mevcut `.venv`
yoksa önce oluştur:

```powershell
py -3.12 -m venv .venv
```

Backend test ve client bağımlılıkları:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,mcp]"
```

Server ortamı yoksa oluştur ve bağımlılığını kur:

```powershell
.\.venv\Scripts\python.exe -m venv .venv-mcp
.\.venv-mcp\Scripts\python.exe -m pip install -r requirements-mcp-server.txt
```

FastMCP'yi backend ortamına kurma. Server ortamına bütün backend'i kurmak da
gerekmez; aşağıdaki komutlar kaynak kodunu `PYTHONPATH` üzerinden bulur.
Paket kurulumları ağ gerektirebilir; demo bir model servisi gerektirmez.

## 1. In-process smoke

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv-mcp\Scripts\python.exe -m llm_benchmark.demo_mcp_smoke
```

Beklenen mesaj: `FastMCP calculator smoke passed.`

Bu kontrol araç kaydı ve çağrısını doğrular. Ayrı process veya stdio bağlantısını
doğrulamaz. `PYTHONPATH` bu terminal oturumunda ayarlanır.

## 2. Gerçek stdio smoke

```powershell
.\.venv\Scripts\python.exe -m llm_benchmark.demo_mcp_stdio_smoke --server-python .\.venv-mcp\Scripts\python.exe
```

Client server'ı kendisi başlatır; önceden ayrı terminalde server açmak gerekmez.
Komutun başındaki Python backend'e, `--server-python` server ortamına aittir.
Client subprocess için `cwd`, `PYTHONPATH` ve `PYTHONUNBUFFERED` ayarlar.

Başarılı mesaj şu biçimdedir:

```text
MCP stdio smoke passed. Protocol: <negotiated-version>; result: 391.
```

Başka dizinden çalıştırılıyorsa `--project-root` ile depo kökünü açıkça ver.
Server'ı doğrudan başlatmak stdin'den protokol mesajı beklemesine yol açar;
bu modda tarayıcı adresi veya HTTP endpoint oluşmaz.

## Testler

```powershell
$env:LLM_BENCHMARK_MCP_PYTHON = (Resolve-Path .\.venv-mcp\Scripts\python.exe).Path
.\.venv\Scripts\python.exe -m pytest -q tests/test_mcp_mapping.py tests/test_mcp_client.py tests/test_mcp_stdio.py
```

`LLM_BENCHMARK_MCP_PYTHON` yoksa stdio testleri skip olur. Yol tanımlı ama dosya
yoksa test fail olur. MCP paketi yoksa SDK gerektiren testler skip olabilir.
PR doğrulamasında stdio testlerinin gerçekten çalıştığını kontrol et.

Negatif argüman testleri string sayı, boolean, sınır dışı sayı, bilinmeyen işlem
ve fazladan alanı server'a doğrudan gönderir. Böylece yalnızca client wrapper'ın
değil server'ın da yanlış girdiyi reddettiği sınanır. Beklenen ret tool error
veya invalid-params protokol hatasıdır; timeout/bağlantı hatası başarı sayılmaz.

Merge öncesi genel kontrol:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
```

Bu environment değişkenleri Hugging Face offline davranışını ayarlar; işletim
sistemi seviyesinde bütün ağ erişimini engellemez. PostgreSQL ve OS bağımlı
skip'leri ayrıca raporla. Test sayısını burada sabit başarı iddiası olarak
tutmak yerine çalıştırılan commit'in sonucunu PR'a kaydet.

## Sınırlar ve davranışlar

- Catalog yalnızca `calculator` kabul eder; başka araç veya devam sayfası reddedilir.
- `next_cursor=None` başka sayfa olmadığını gösterir ve kabul edilir.
- Schema byte bütçesi 65.536; desteklenen keyword kümesi sınırlıdır.
- Mapping yalnızca structured JSON object çıktısını destekler.
- Varsayılan sonuç bütçesi 65.536 UTF-8 byte'tır; decode sonrası uygulanır,
  transport mesajının veya peak memory kullanımının sınırı değildir.
- Modelin `call_id` değeri `ToolResult` üzerinde korunur.
- Ham hata payload'u başarısız `ToolResult` çıktısına kopyalanmaz.
- Launch yolları güvenilir yerel geliştirici girdisidir; API komut arayüzü değildir.
- Server stdout'una debug `print()` eklenmemeli; stdio protokolüne ayrılmıştır.
- Smoke'ta 30 saniyelik dış bütçe ve 10 saniyelik istek timeout'ları vardır.
  Bunlar production process recovery veya keyfî Python kodunu zorla durdurma garantisi değildir.

## Sorun giderme

| Belirti | Kontrol |
|---|---|
| `No module named fastmcp` | Server `.venv-mcp` Python'ıyla başlatılıyor mu? |
| Proje modülü bulunamıyor | In-process smoke için `PYTHONPATH`, stdio için `--project-root` doğru mu? |
| `single page catalog` | `next_cursor is not None` koşulu doğru mu; server gerçekten devam sayfası mı sunuyor? |
| Schema reddediliyor | Server'ın keşfedilen input schema'sı desteklenen profile uyuyor mu? |
| `ExceptionGroup` | En içteki exception'ı oku; dış task-group katmanları kök nedeni sarmalıyor olabilir |
| Testler skip | Server Python environment değişkeni ve backend MCP extra kurulmuş mu? |

## Sonraki teslimler

1. Bu bağlantı temelini mapping, discovery, server ve stdio testleriyle PR olarak kapat.
2. Ayrı PR'da senkron `ToolExecutor` / async MCP session köprüsünü kur.
   Session sahibi üst orchestration olsun; her çağrıda yeni session açılmasın.
3. Timeout, transport ve protocol hatalarını ayır; kapanış ve server çökmesi
   senaryolarını test et. Yerel/MCP suite değerlendirmelerini karşılaştır.
4. Sonraki PR'da runtime seçimini `tool_run_config.py` ve `tool_run_service.py`
   üzerinden artifact/provenance akışına bağla.

İlk PR, MCP'nin benchmark'a tamamen entegre olduğu iddiasını taşımamalıdır.
