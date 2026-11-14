# ZIP upload → chunks + embeddings → ChromaDB release

## Gebruik via GitHub

1. Upload één ZIP naar [`standalone-models/ingestion target/`](../standalone-models/ingestion%20target/).
   De map is leeg behalve `.gitkeep`; dat bestand is nodig om een lege map in Git te tonen.
2. Open **Actions** en kies de gewenste workflow:
   - [Ingest ZIP — BGE-M3](https://github.com/Dosa42/Embedding-models/actions/workflows/ingest-bge-m3.yml)
   - [Ingest ZIP — E5 base v2](https://github.com/Dosa42/Embedding-models/actions/workflows/ingest-e5-base-v2.yml)
   - [Ingest ZIP — E5 large v2](https://github.com/Dosa42/Embedding-models/actions/workflows/ingest-e5-large-v2.yml)
3. Kies **Run workflow**, op de branch waarin je ZIP staat.
4. Vul eventueel de exacte ZIP-bestandsnaam in. Laat dit leeg als er precies één ZIP
   in de map staat. Bij meerdere ZIPs is een bestandsnaam verplicht.
5. Laat standaard **q8**, **384 tokens**, **64 overlap** en **batch 4** staan, of pas ze aan.
6. Na een geslaagde run staat de downloadlink in de samenvatting en onder
   [Releases](https://github.com/Dosa42/Embedding-models/releases). Releases heten
   `ingestion-<model>-<run-id>-<attempt>` en vervangen bestaande releases niet.

De workflows starten alleen handmatig. Het uploaden van een ZIP start dus geen
verwerking. De originele ZIP blijft in de uploadmap staan. Uitpakken gebeurt in
een tijdelijke map op de runner; documenten en output worden niet teruggecommit.

### ZIPs uploaden

GitHub's browser upload ondersteunt bestanden tot 25 MiB. Voor grotere ZIPs,
gebruik Git met Git LFS en track de input-ZIPs:

```sh
git lfs install
git lfs track 'standalone-models/ingestion target/*.zip'
# Kopieer dataset.zip naar standalone-models/ingestion target/
git add .gitattributes 'standalone-models/ingestion target/dataset.zip'
git commit -m 'Upload ingestion dataset'
git push
```

Gebruik een gewone bestandsnaam, bijvoorbeeld `mijn documenten.zip`, zonder
submappen, komma's of globtekens (`* ? [ ]`). Met één ZIP kan de workflow de naam
zelf kiezen. De runner haalt alleen het gekozen model en de gekozen LFS-input op.

## Uitvoer

Iedere databaserelease bevat:

- `chunked-chromadb-<model>-<dtype>.zip`: volledige databasebundel.
- `SHA256SUMS.txt`: checksum van de databasebundel.
- `manifest.json`: aantallen, modelversie, inputchecksum en verslag per bronbestand.

De ZIP bevat:

| Bestand | Inhoud |
| --- | --- |
| `chroma_db/` | Complete native ChromaDB-opslagmap met collectie `chunks`, bronmetadata en cosine-vectorindex |
| `chunks.jsonl` | Leesbare tekstchunks met bronbestand, segment, paginanummer waar beschikbaar en offsets |
| `manifest.json` | Herkomst, instellingen, versies, checksums en overgeslagen/lege bestanden |
| `ingestion/` + `requirements.txt` | Hulpmiddel om semantisch in de database te zoeken |
| `README.txt` | Opslagformaat en gebruik |

Dit is een **native ChromaDB-database**. Bewaar de hele map `chroma_db/` samen.
Chroma beheert intern metadata en indexbestanden, waaronder `chroma.sqlite3`.
Een los `.db`-bestand of het hernoemen van interne bestanden is geen geldige
Chroma-export. De release-ZIP bevat de complete opslagmap, zodat niets hoeft
te worden samengevoegd of opnieuw geïndexeerd.

**q8 blijft de standaard voor het ONNX-model.** De geproduceerde embeddings
worden als genormaliseerde float32-vectoren aan Chroma gegeven. Dit is geen
8-bit quantisatie van de Chroma-index. Modelidentiteit, precisie, pooling en
prefix staan in de collectiemetadata en in `manifest.json`.

Na uitpakken, met Python **3.12**:

```sh
python -m pip install -r requirements.txt
```

Direct openen met Chroma:

```python
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("chunks", embedding_function=None)
print(collection.count())
# Gebruik query_embeddings van hetzelfde model, of de meegeleverde zoektool.
client.close()
```

Semantisch zoeken met het originele, uitgepakte model:

```sh
python -m ingestion.search \
  --database chroma_db \
  --model-dir /pad/naar/Xenova/bge-m3 \
  --query 'Waar gaat dit document over?' --top-k 5
```

De zoektool controleert model/tokenizer-identiteit, gebruikt de juiste prefix en
zoekt via Chroma's cosine-index. Chroma-filters op metadata en `where_document`
blijven beschikbaar. De JSONL-export kan ook in andere systemen worden gebruikt.
Gebruik de meegeleverde Chroma-versie voor het openen van deze export; een andere
versie kan de database migreren.

De schrijver sluit voordat export begint. Een tweede proces heropent de database,
controleert recordaantallen, vectordimensies, normalisatie en een indexquery, en
sluit daarna opnieuw. Pas dan wordt de complete opslagmap verpakt en gepubliceerd.

## Documenten en chunking

Ondersteund: UTF-8-tekst/code, UTF-16 met BOM, Markdown, CSV/TSV, JSON/JSONL,
YAML/XML, HTML, PDF met tekstlaag, DOCX (body-paragrafen en tabellen), en gangbare
programmeertalen. HTML-script/style-inhoud wordt niet meegenomen. Documenten
worden nooit als programma uitgevoerd. Geneste ZIPs worden niet verder uitgepakt.

De tokenizer van het gekozen model bepaalt de chunkgrenzen en overlap. Elke chunk
wordt opnieuw gecontroleerd op de modelcontext; er is geen stille afkapping.
E5-documenten krijgen `passage: ` en zoekvragen `query: `, met masked mean pooling.
BGE-M3 gebruikt CLS-pooling. Alle vectors zijn L2-genormaliseerd. E5-v2 is bedoeld
voor Engels; BGE-M3 ondersteunt meertalige tekst.

Bronpaden, bronchecksums en paginanummers voor PDF blijven bewaard. Offsets en
regelnummers verwijzen naar de **geëxtraheerde tekst**, niet naar PDF-byteposities.
PDF's zonder tekstlaag, versleutelde documenten, onleesbare tekst en niet-ondersteunde
bestanden verschijnen in het verslag; er is geen OCR. Een volledig lege/onleesbare
invoer levert een fout op en publiceert geen lege database.

## Instellingen en grenzen

- Standaard: q8 op CPU, chunk 384 tokens, overlap 64, batch 4, 2 CPU-threads.
- FP16 is ook selecteerbaar, maar gebruikt meer geheugen; q8 is de standaard.
- `overlap` moet kleiner zijn dan `chunk_tokens`; batch is 1–16.
- Het tokenbudget is inclusief modelcontextbeperkingen; E5-v2 maximaal 512 tokens
  inclusief prefix/speciale tokens. BGE-M3 maximaal 8192. Grote chunks kosten
  veel meer geheugen; begin met 384.
- ZIP: maximaal 20.000 entries, 100 MiB per uitgepakt bestand, 2 GiB totaal uitgepakt.
- Maximaal 5 miljoen tekens per tekstsegment, 20 miljoen per document en 100.000 chunks.
- Runner-timeout: 180 minuten. Verdeel grote datasets over meerdere ZIPs/runs.
- Elke release-asset blijft kleiner dan 2 GiB; een te grote output wordt afgewezen.

Fouten tijdens embeddings of integriteitscontrole stoppen de publicatie. Een fout
tijdens uploaden kan een ongepubliceerde draft-release achterlaten. Een database
wordt pas zichtbaar als alle releasebestanden zijn geüpload. Repositoryzichtbaarheid
geldt ook voor de input en database-releases.

## Lokaal draaien en testen

```sh
python3.12 -m pip install -r ingestion/requirements.txt
python3.12 -m unittest discover -s tests -p 'test_ingestion.py' -v
python3.12 -m ingestion.build \
  --input documenten.zip --model-dir models/Xenova/bge-m3 \
  --model bge-m3 --dtype q8 --output dist/mijn-database
```

De outputmap moet leeg zijn. Lokale verwerking gebruikt dezelfde gelockte modellen,
checksums, tokenizer, chunking en ChromaDB-opslag als de workflows.
