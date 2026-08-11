# V1/MVP varsayımları

- Sonuç deposu V1'de koşu başına append-friendly JSONL ve türetilmiş JSON
  özettir; SQLite/PostgreSQL sonraki dilime bırakılmıştır.
- Config tek YAML dosyasıdır. Öncelik built-in Pydantic defaults, YAML ve açık
  `--set` override sırasındadır. Katmanlı config ve migration V2 konusudur.
- `schema_version` yalnızca `1` değerini kabul eder; bilinmeyen alanlar hatadır.
- Fixture verisi sentetiktir ve yalnızca yazılım doğrulaması içindir. Bir
  benchmark skoru olarak yorumlanmaz.
- MMLU-Pro `revision` alanı zorunludur. `smoke` 14, `poc` mümkün olduğunda
  kategori başına 10 örnek seçer; `full` filtrelenmiş test split'inin tamamıdır.
- MMLU-Pro loader seçenekleri upstream `options`, cevabı `answer`, kimliği
  varsa `question_id`, yoksa kaynak sıra numarası alanından normalize eder.
- Mock provider örnek bazlı deterministik senaryolar kullanır ve gerçek sleep
  yapmaz. Ölçülen sahte latency provider sonucundan gelir.
- V1 gerçek retry uygulamaz; `attempt_count=1` ve logical latency attempt
  latency'ye eşittir. Retry yürütme ikinci dilimin işidir.
- Ana latency özeti yalnızca request'i başarılı sonuçların logical latency
  değerlerini kullanır. Başarısız latency ayrı `failure_latency_*` alanlarındadır.
- `incorrect_count`, parse edilebilen yanlış cevapları ifade eder;
  `unparseable` ve `request_failed` ayrı sayılır. Ana accuracy paydası tüm
  planlanan örneklerdir.
- Maliyet V1'de hesaplanmaz ve `estimated_cost` null kalır.
- Token bilgisi yoksa null tutulur. Token toplamları yalnızca bilinen usage
  değerlerini toplar ve eksik kayıt sayısı ayrıca raporlanır.
- Parser A-J yerine her sorunun gerçek `allowed_labels` kümesini kullanır.
- Log-probability, self-consistency, sayısal tolerans, streaming/TTFT, gerçek
  provider, veritabanı, web UI ve orchestration sonraki dilimlerdir.
- Varsayılan çıktı deposu `outputs` olup üretilen koşular Git tarafından yok
  sayılır.
