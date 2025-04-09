## Empowering AI Models: Fine-Tuning RAG Data and Human Feedback

## Yapay Zeka Modellerini Güçlendirme: RAG Verilerini İnce Ayarlama ve İnsan Geri Bildirimi (Empowering AI Models: Fine-Tuning RAG Data and Human Feedback)

Bir organizasyon sürekli olarak RAG (Retrieval-Augmented Generation) verilerinin hacmini artırdığında, parametrik olmayan verilerin (LLM üzerinde önceden eğitilmemiş) eşiğine ulaşacaktır. Bu noktada, biriken RAG verilerinin kitlesi, depolama maliyetleri, erişim kaynakları ve üretken yapay zeka modellerinin kapasitesi ile ilgili sorunlar yaratabilir. Ayrıca, önceden eğitilmiş bir üretken yapay zeka modeli, bir kesinti tarihine kadar eğitilir. Model, ertesi günden itibaren yeni bilgiyi görmezden gelir. Bu, bir kullanıcının kesinti tarihinden sonra yayınlanan bir gazete baskısının içeriği üzerinde bir sohbet modeliyle etkileşim kurmasının imkansız olacağı anlamına gelir. İşte o zaman erişim, RAG tarafından yönlendirilen içerik sağlamada kilit bir rol oynar.

## RAG Verilerini İnce Ayarlama Mimarisi (Fine-Tuning RAG Data Architecture)

Bu bölümde, önce RAG verilerini ince ayarlayarak azaltma mimarisini inceleyeceğiz. İnsan geri bildirim faktörünü de içeren kullanıma hazır belgeler içeren bir veri kümesine odaklanacağız. Parametrik olmayan verileri OpenAI modelinde parametrik, ince ayarlı verilere nasıl dönüştüreceğimizi göstereceğiz.

### Veri Kümesini Hazırlama (Preparing the Dataset)

Önceki bölümdeki veri kümesini indirecek ve ince ayar için iyi biçimlendirilmiş istem (prompt) ve tamamlama (completion) çiftlerine JSONL formatında dönüştüreceğiz.

```python
import json

# Veri kümesini yükleme
with open('data.json', 'r') as f:
    data = json.load(f)

# İstem ve tamamlama çiftlerini oluşturma
prompt_completion_pairs = []
for item in data:
    prompt = item['prompt']
    completion = item['completion']
    prompt_completion_pairs.append({'prompt': prompt, 'completion': completion})

# JSONL formatında kaydetme
with open('data.jsonl', 'w') as f:
    for pair in prompt_completion_pairs:
        json.dump(pair, f)
        f.write('\n')
```

Bu kod, `data.json` dosyasını yükler, istem ve tamamlama çiftlerini oluşturur ve `data.jsonl` dosyasına JSONL formatında kaydeder.

### OpenAI Modelini İnce Ayarlama (Fine-Tuning the OpenAI Model)

İnce ayar için maliyet etkin bir OpenAI modeli olan GPT-4o-mini'yi kullanacağız.

```python
import os
import openai

# OpenAI API anahtarını ayarlama
openai.api_key = os.getenv('OPENAI_API_KEY')

# Veri kümesini yükleme
with open('data.jsonl', 'r') as f:
    data = f.readlines()

# İnce ayar işlemini başlatma
response = openai.File.create(
    file=data,
    purpose='fine-tune'
)

# İnce ayar modelini oluşturma
response = openai.FineTune.create(
    training_file=response.id,
    model='gpt-4o-mini'
)

# İnce ayar işlemini izleme
response = openai.FineTune.retrieve(response.id)
print(response.status)
```

Bu kod, OpenAI API anahtarını ayarlar, veri kümesini yükler, ince ayar işlemini başlatır ve ince ayar modelini oluşturur.

### İnce Ayar Modelini Test Etme (Testing the Fine-Tuned Model)

İnce ayar modelini veri kümemizde test edeceğiz.

```python
# İnce ayar modelini yükleme
model = openai.Model.retrieve('gpt-4o-mini')

# Test verilerini yükleme
with open('test_data.json', 'r') as f:
    test_data = json.load(f)

# Test işlemini başlatma
for item in test_data:
    prompt = item['prompt']
    completion = model.generate(prompt)
    print(completion)
```

Bu kod, ince ayar modelini yükler, test verilerini yükler ve test işlemini başlatır.

## Sonuç (Conclusion)

Bu bölümde, RAG verilerini ince ayarlayarak azaltma mimarisini inceledik. İnsan geri bildirim faktörünü de içeren kullanıma hazır belgeler içeren bir veri kümesine odaklandık. Parametrik olmayan verileri OpenAI modelinde parametrik, ince ayarlı verilere nasıl dönüştüreceğimizi gösterdik. Ayrıca, ince ayar modelini test ettik ve OpenAI'ın metrik arayüzünü kullanarak teknik metriklerimizi izledik.

## Önemli Noktalar (Key Points)

* RAG verilerini ince ayarlayarak azaltma mimarisini inceledik.
* İnsan geri bildirim faktörünü de içeren kullanıma hazır belgeler içeren bir veri kümesine odaklandık.
* Parametrik olmayan verileri OpenAI modelinde parametrik, ince ayarlı verilere nasıl dönüştüreceğimizi gösterdik.
* İnce ayar modelini test ettik ve OpenAI'ın metrik arayüzünü kullanarak teknik metriklerimizi izledik.

## Konu ile İlgili Teknik Terimler (Technical Terms)

* RAG (Retrieval-Augmented Generation)
* LLM (Large Language Model)
* JSONL (JSON Lines)
* GPT-4o-mini (Generative Pre-trained Transformer 4 optimized for mini tasks)

---

## The architecture of fine-tuning static RAG data

## Statik RAG Verilerinin İnce Ayar Mimarisinin (Fine-Tuning) İncelenmesi

Bu bölümde, RAG (Retrieval Augmented Generation) verilerinin yönetilebilir bir eşiği aştığında non-parametrik (non-parametric) kullanımını sorguluyoruz. Bu durum, Bölüm 1'de "RAG vs İnce Ayar RAG vs İnce Ayar" bölümünde anlatılan ve bir eşiğin (threshold) tanımlandığı "Neden Retrieval Augmented Generation?" bölümünde belirtilen ilkeye dayanmaktadır.

## RAG Veri Eşiği ve İnce Ayar

Şekil 9.1, bu ilkeyi bu bölüme uyarlamaktadır:
### Şekil 9.1: RAG Verileri için İnce Ayar Eşiği Ulaşıldı

Şekil 9.1'de, statik veriler için işleme (D2) ve depolama (D3) eşikleri, RAG veri ortamındaki dinamik verilere kıyasla gösterilmektedir. Eşik, her proje ve parametreye bağlıdır, örneğin:
*   RAG verilerinin işlenmesi gereken hacim (The volume of RAG data to process): Embedding verileri insan ve makine kaynakları gerektirir. Verileri embed etmesek bile, uzun bir süre boyunca stabil kalan statik verileri biriktirmek anlamsızdır.
*   Depolanacak ve alınacak RAG verilerinin hacmi (The volume of RAG data to store and retrieve): Bir noktada, verileri yığmaya devam edersek, çoğu örtüşebilir. Alımlamalar kaynak gerektirir (The retrievals require resources): Sistem açık kaynaklı olsa bile, yönetilecek kaynakların sayısı artmaktadır.

Diğer faktörler de her proje için rol oynayabilir. Neden olursa olsun, RAG veri eşiğine ulaştığımızda ince ayar (fine-tuning) iyi bir çözüm olabilir.

## İnce Ayar için Kod Örneği

İnce ayar yapmak için aşağıdaki kod örneğini kullanabiliriz:
```python
# Import gerekli kütüphaneler
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Model ve tokenizer yükleme
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Veri yükleme ve hazırlama
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Verileri tokenize etme
train_encodings = tokenizer(list(train_data["text"]), truncation=True, padding=True)
test_encodings = tokenizer(list(test_data["text"]), truncation=True, padding=True)

# Tensor oluşturma
train_tensors = torch.tensor(train_encodings["input_ids"])
test_tensors = torch.tensor(test_encodings["input_ids"])

# İnce ayar için model hazırlama
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_tensors = train_tensors.to(device)
test_tensors = test_tensors.to(device)

# İnce ayar yapma
# Modeli eğitme moduna alma
model.train()

# Optimizer ve loss function tanımlama
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Eğitim döngüsü
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(train_tensors, labels=torch.tensor(train_data["label"]).to(device))
    loss = loss_fn(outputs, torch.tensor(train_data["label"]).to(device))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Modeli değerlendirme moduna alma
model.eval()

# Test verileri üzerinde değerlendirme yapma
with torch.no_grad():
    outputs = model(test_tensors)
    _, predicted = torch.max(outputs.scores, dim=1)
    accuracy = (predicted == torch.tensor(test_data["label"]).to(device)).sum().item() / len(test_data)
    print(f"Test Accuracy: {accuracy:.4f}")
```
### Kod Açıklaması

Yukarıdaki kod, BERT (Bidirectional Encoder Representations from Transformers) modelini kullanarak ince ayar yapma örneğini göstermektedir.

1.  İlk olarak, gerekli kütüphaneler (`pandas`, `torch`, `transformers`) içe aktarılır.
2.  `bert-base-uncased` modeli ve tokenizer yüklenir.
3.  Eğitim ve test verileri `train.csv` ve `test.csv` dosyalarından okunur.
4.  Veriler tokenize edilir ve tensorlere dönüştürülür.
5.  Model, cihaz (GPU veya CPU) üzerine taşınır ve veriler de aynı cihaza taşınır.
6.  Model eğitme moduna alınır ve optimizer ile loss function tanımlanır.
7.  Eğitim döngüsü içerisinde, model sıfırlanır, çıktıları hesaplanır, loss hesaplanır, geriye doğru yayılım yapılır ve optimizer adım atar.
8.  Eğitimden sonra, model değerlendirme moduna alınır ve test verileri üzerinde değerlendirme yapılır.

Bu kod örneği, RAG verileri için ince ayar yapma sürecini göstermektedir. İnce ayar, önceden eğitilmiş bir modelin belirli bir görev için daha da eğitilmesini sağlar. Bu sayede, modelin performansı artırılabilir.

---

## The RAG ecosystem

## RAG Ekosistemi (The RAG Ecosystem)

Bu bölümde, 1. Bölüm'de anlatılan RAG ekosistemine geri döneceğiz ve bu bölüm için gerekli olan bileşenlere odaklanacağız. Aşağıdaki şekil, ince ayar bileşenlerini renkli olarak ve ihtiyacımız olmayan bileşenleri gri renkte göstermektedir:

## İnce Ayar Bileşenleri (Fine-Tuning Components)

RAG ekosisteminin ince ayar bileşenlerinin ana özellikleri aşağıdaki noktalarda özetlenebilir:
- Veri toplama (D1) ve hazırlama (D2) : İnsan tarafından oluşturulan SciQ sert bilim veri setini indirecek ve işleyeceğiz. Veri seti https://huggingface.co/datasets/sciq adresinde mevcuttur.
- İnsan geri bildirimi (E2) : SciQ veri seti insan kontrolü altında olduğundan, güvenilir insan geri bildiriminin RAG veri setlerinin hacmini azaltmak için nasıl ince ayar yapılabileceğini simüle edebiliriz.
- İnce ayar (T2) : Maliyet etkin bir OpenAI modeli olan GPT-4o-mini'yi ince ayar yapacağız.
- Prompt mühendisliği (G3) ve üretim ve çıktı (G4) : OpenAI tarafından önerilen şekilde prompt'leri mühendislik yapacağız ve çıktıyı görüntüleyeceğiz.
- Metrikler (E1) : OpenAI'ın Metrikler arayüzünün ana özelliklerine bakacağız.

### SciQ Veri Setini İndirme ve İşleme

SciQ veri setini indirmek ve işlemek için aşağıdaki Python kodunu kullanacağız:
```python
import pandas as pd
from datasets import load_dataset

# SciQ veri setini yükle
dataset = load_dataset("sciq")

# Veri setini DataFrame'e çevir
df = pd.DataFrame(dataset["train"])

# Veri setini incele
print(df.head())
```
Bu kod, `datasets` kütüphanesini kullanarak SciQ veri setini yükler ve `pandas` kütüphanesini kullanarak DataFrame'e çevirir. Daha sonra, veri setinin ilk birkaç satırını yazdırır.

### GPT-4o-mini Modelini İnce Ayar Yapma

GPT-4o-mini modelini ince ayar yapmak için aşağıdaki Python kodunu kullanacağız:
```python
import openai

# OpenAI API anahtarını ayarla
openai.api_key = "YOUR_API_KEY"

# GPT-4o-mini modelini ince ayar yap
model = openai.Model("gpt-4o-mini")

# İnce ayar parametrelerini ayarla
params = {
    "temperature": 0.7,
    "max_tokens": 1024,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0
}

# İnce ayar yap
response = model.edit(
    input="YOUR_INPUT_TEXT",
    params=params
)

# Çıktıyı yazdır
print(response.choices[0].text.strip())
```
Bu kod, OpenAI API anahtarını ayarlar ve GPT-4o-mini modelini ince ayar yapar. Daha sonra, ince ayar parametrelerini ayarlar ve ince ayar yapar. Son olarak, çıktıyı yazdırır.

### Prompt Mühendisliği ve Üretim ve Çıktı

Prompt mühendisliği ve üretim ve çıktı için aşağıdaki Python kodunu kullanacağız:
```python
import openai

# OpenAI API anahtarını ayarla
openai.api_key = "YOUR_API_KEY"

# Prompt'u mühendislik yap
prompt = "YOUR_PROMPT_TEXT"

# GPT-4o-mini modelini kullanarak çıktı üret
response = openai.Completion.create(
    engine="gpt-4o-mini",
    prompt=prompt,
    max_tokens=1024,
    temperature=0.7
)

# Çıktıyı yazdır
print(response.choices[0].text.strip())
```
Bu kod, OpenAI API anahtarını ayarlar ve prompt'u mühendislik yapar. Daha sonra, GPT-4o-mini modelini kullanarak çıktı üretir ve çıktıyı yazdırır.

### Metrikler

OpenAI'ın Metrikler arayüzünün ana özelliklerine bakmak için aşağıdaki Python kodunu kullanacağız:
```python
import openai

# OpenAI API anahtarını ayarla
openai.api_key = "YOUR_API_KEY"

# Metrikleri al
metrics = openai.Metric.list()

# Metrikleri yazdır
for metric in metrics:
    print(metric.name)
```
Bu kod, OpenAI API anahtarını ayarlar ve metrikleri alır. Daha sonra, metrikleri yazdırır.

---

## Installing the environment

## Ortamın Kurulumu (Installing the Environment)
Yapay zeka (AI) ve çapraz platform bağımlılık çakışmalarının (cross-platform dependency conflicts) hızlı evrimi ile ortam kurulumu karmaşık hale gelmiştir. Bu nedenle, mümkün olduğunda paket sürümlerini donduracağız (freeze the package versions). Bu program için, GitHub'daki Chapter09 dizininde bulunan `Fine_tuning_OpenAI_GPT_4o_mini.ipynb` not defterini (notebook) açın.

### OpenAI API Anahtarının Alınması (Retrieving the OpenAI API Key)
Program ilk olarak OpenAI API anahtarını alır:
```python
# API anahtarınızı bir dosyadan alabilirsiniz (1)
# veya manuel olarak girebilirsiniz (2)
# Bu hücreyi (cell) API anahtarınızı manuel olarak girmek istiyorsanız yorumlayın (comment).

#(1) API Anahtarını bir dosyadan alın
# API anahtarınızı bir dosyada saklayın ve okuyun (not defterinde doğrudan yazabilirsiniz ancak yanınızdaki biri tarafından görülebilir)
from google.colab import drive
drive.mount('/content/drive')
f = open("drive/MyDrive/files/api_key.txt", "r")
API_KEY = f.readline()
f.close()
```
Bu kod, Google Colab'da (Google Colab) `drive` modülünü kullanarak `api_key.txt` dosyasından OpenAI API anahtarını okur.

### OpenAI Kütüphanesinin Kurulumu (Installing the OpenAI Library)
Daha sonra `openai` kütüphanesini kurar ve API anahtarını ayarlarız:
```python
try:
    import openai
except:
    !pip install openai==1.42.0
import openai

#(2) API anahtarınızı manuel olarak girin
# API_KEY değişkenini anahtarınızla değiştirin.
import os
os.environ['OPENAI_API_KEY'] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
```
Bu kod, `openai` kütüphanesini kurar ve API anahtarını ortam değişkeni (environment variable) olarak ayarlar.

### jsonlines Kütüphanesinin Kurulumu (Installing the jsonlines Library)
Şimdi `jsonlines` kütüphanesini kurarız:
```bash
!pip install jsonlines==4.0.0
```
Bu kütüphane, JSONL (JSON Lines) verilerini oluşturmak için kullanılır.

### datasets Kütüphanesinin Kurulumu (Installing the datasets Library)
Daha sonra `datasets` kütüphanesini kurarız:
```bash
!pip install datasets==2.20.0
```
Bu kütüphane, veri kümelerini (dataset) indirmek ve işlemek için kullanılır.

### Ortamın Hazır Hale Getirilmesi (Preparing the Environment)
Chapter 8'de (Chapter 8) anlatılan `datasets` kütüphanesinin kurulumu sırasında ortaya çıkan bağımlılık çakışmalarına (dependency conflicts) dikkat edin. Bazı kurulum sorunları ortaya çıkabilir ancak veri kümesi yine de indirilecektir. Lider platformların sürekli olarak paketlerini güncellemesi ve önceden kurulmuş ortamlarla (pre-installed environments) çakışmalar yaratması beklenmelidir. Bu program için özel bir ortam oluşturabilirsiniz. Diğer programlarınızın diğer paket kısıtlamaları (package constraints) nedeniyle sorunlar yaşayabileceğini unutmayın.

Artık veri kümesini hazırlamaya hazırız.

Önemli noktalar:
* Paket sürümlerini dondurmak (freeze the package versions) için belirli sürümleri kullanın.
* OpenAI API anahtarını güvenli bir şekilde saklayın ve alın.
* Gerekli kütüphaneleri (`openai`, `jsonlines`, `datasets`) kurun.
* Ortamın hazır hale getirilmesi için Chapter 8'de anlatılan adımları takip edin.

---

## 1. Preparing the dataset for fine-tuning

## 1. İnce Ayar (Fine-Tuning) için Veri Setini Hazırlama

İnce ayar (fine-tuning) bir OpenAI modelini eğitmek için özenli bir hazırlık gerektirir; aksi takdirde, ince ayar görevi başarısız olur. Bu bölümde, aşağıdaki adımları gerçekleştireceğiz:

* Hugging Face'den veri setini indirin ve sütunlarını işleyerek hazırlayın.
* Veri setini JSONL formatında bir JSON dosyasına akıtın (stream).

### Veri Setini İndirme ve Hazırlama

Program, veri setini indirerek başlar. Bu işlem için `datasets` kütüphanesini kullanacağız.

```python
import pandas as pd
import json
from datasets import load_dataset

# Veri setini Hugging Face'den indirin
dataset = load_dataset("your_dataset_name")
```

* `load_dataset` fonksiyonu, Hugging Face'den belirtilen veri setini indirir.
* `your_dataset_name` yerine kullanmak istediğiniz veri setinin adını yazın.

### Veri Setini JSONL Formatına Dönüştürme

Veri setini JSONL formatına dönüştürmek için aşağıdaki kodu kullanacağız:

```python
# Veri setini JSONL formatında bir JSON dosyasına akıtın
with open('data.jsonl', 'w') as f:
    for example in dataset['train']:
        json.dump(example, f)
        f.write('\n')
```

* `open` fonksiyonu, `data.jsonl` adında bir dosya oluşturur ve yazma modunda açar.
* `json.dump` fonksiyonu, her bir örnek (example) için JSON formatında yazar.
* `f.write('\n')` ifadesi, her bir örnek arasına bir satır sonu ekler.

### Örnek Kod

Tüm kodları bir araya getirdiğimizde:

```python
import pandas as pd
import json
from datasets import load_dataset

# Veri setini Hugging Face'den indirin
dataset = load_dataset("your_dataset_name")

# Veri setini JSONL formatında bir JSON dosyasına akıtın
with open('data.jsonl', 'w') as f:
    for example in dataset['train']:
        json.dump(example, f)
        f.write('\n')
```

* Bu kod, veri setini indirir, işler ve JSONL formatında bir JSON dosyasına yazar.

### Önemli Noktalar

* Veri setini indirmek için `datasets` kütüphanesini kullanın.
* Veri setini JSONL formatına dönüştürmek için `json.dump` fonksiyonunu kullanın.
* Her bir örnek arasına bir satır sonu ekleyin.

### Açıklamalar

* `load_dataset` fonksiyonu, Hugging Face'den veri setini indirir.
* `json.dump` fonksiyonu, Python nesnelerini JSON formatında yazar.
* JSONL formatı, her bir satırın bağımsız bir JSON nesnesi olduğu bir metin dosyasıdır.

---

## 1.1. Downloading and visualizing the dataset

## 1.1. Veri Kümesini İndirme ve Görselleştirme (Downloading and Visualizing the Dataset)

Bu bölümde, daha önce 8. Bölüm'de gömülü (embedded) olan SciQ veri kümesini indireceğiz. Binlerce belgeyi gömmek (embedding) zaman ve kaynak gerektiren bir işlemdir. Bu bölümde, veri kümesini indireceğiz, ancak bu kez gömmeyeceğiz (embed). OpenAI modelinin ince ayar (fine-tuning) sırasında bu işlemi bizim için yapmasına izin vereceğiz.

### Veri Kümesini İndirme ve Filtreleme

İlk olarak, gerekli kütüphaneleri içe aktaralım (importing libraries):
```python
from datasets import load_dataset
import pandas as pd
```
Şimdi, Hugging Face'den SciQ veri kümesini indirelim:
```python
dataset_view = load_dataset("sciq", split="train")
```
Bu kod, Hugging Face'den "sciq" veri kümesini indirir ve "train" bölümünü `dataset_view` değişkenine atar.

Daha sonra, veri kümesini filtreleyerek (filtering) sadece doğru cevap ve destek metni (support text) olan soruları dahil edelim:
```python
filtered_dataset = dataset_view.filter(lambda x: x["support"] != "" and x["correct_answer"] != "")
```
Bu kod, `dataset_view` veri kümesini filtreleyerek, destek metni ve doğru cevap olan soruları `filtered_dataset` değişkenine atar.

Filtrelenmiş soruların sayısını yazdıralım:
```python
print("Destek metni olan soruların sayısı: ", len(filtered_dataset))
```
Bu kod, filtrelenmiş soruların sayısını yazdırır. Çıktı olarak 10,481 kayıt görürüz:
```
Destek metni olan soruların sayısı: 10481
```
### Veri Kümesini DataFrame'e Dönüştürme

Şimdi, filtrelenmiş veri kümesini bir DataFrame'e dönüştürelim:
```python
df_view = pd.DataFrame(filtered_dataset)
```
Bu kod, `filtered_dataset` veri kümesini bir DataFrame'e dönüştürür ve `df_view` değişkenine atar.

Daha sonra, yanlış cevapları içeren sütunları (distractor columns) kaldıralım:
```python
columns_to_drop = ['distractor3', 'distractor1', 'distractor2']
df_view = df_view.drop(columns=columns_to_drop)
```
Bu kod, `df_view` DataFrame'inden yanlış cevapları içeren sütunları kaldırır.

Son olarak, DataFrame'in ilk birkaç satırını görüntüleyelim:
```python
df_view.head()
```
Bu kod, `df_view` DataFrame'inin ilk birkaç satırını görüntüler. Çıktı olarak, ihtiyacımız olan üç sütunu görürüz:

## Figure 9.3: Output displaying three columns

İhtiyacımız olan sütunlar: `question` (soru), `correct_answer` (doğru cevap) ve `support` (destek metni). `question` sütunu prompt (istem) haline gelecektir. `correct_answer` ve `support` sütunları ise completion (tamamlama) için kullanılacaktır.

Artık veri kümesini incelediğimize göre, veri kümesini doğrudan bir JSON dosyasına akıtabiliriz (streaming).

---

## 1.2. Preparing the dataset for fine-tuning

## 1.2. İnce Ayar için Veri Kümesini Hazırlama (Preparing the Dataset for Fine-Tuning)

İnce ayar (fine-tuning) için kullanılan tamamlama modelini (completion model) eğitmek için, JSONL formatında (JSON Lines format) çok hassas bir JSON dosyası yazmamız gerekir. Veri kümesini, 1.1. Veri Kümesini İndirme ve Görselleştirme bölümünde yaptığımız gibi indirip işliyoruz, bu bölümde veri kümesini ince ayardan önce kontrol etmeniz önerilir.

### Veri Kümesini JSONL Formatında Yazma

GPT-4o-mini için mesajları JSONL formatında yazıyoruz:
```python
# JSON satırları dosyası için veri öğelerini hazırla (Prepare the data items for JSON lines file)
items = []
for idx, row in df.iterrows():
    detailed_answer = row['correct_answer'] + " Açıklama: " + row['support']
    items.append({
        "messages": [
            {"role": "system", "content": "Bilim sorusu verildiğinde, doğru cevabı ayrıntılı bir açıklama ile verin."},
            {"role": "user", "content": row['question']},
            {"role": "assistant", "content": detailed_answer}
        ]
    })
```
İlk olarak, doğru cevap (`'correct_answer'`) ve destekleyici (`'support'`) açıklama ile ayrıntılı cevabı (`detailed_answer`) tanımlarız. Daha sonra GPT-4o-mini modeli için mesajları (`"messages"`) tanımlarız:

*   `{"role": "system", "content": ...}`: Dil modeline (language model) ilk talimatı verir, bilim sorularına ayrıntılı cevaplar vermesini söyler.
*   `{"role": "user", "content": row['question']}`: Kullanıcının sorduğu soruyu temsil eder, DataFrame'in `question` sütunundan alınır.
*   `{"role": "assistant", "content": detailed_answer}`: Asistanın cevabını temsil eder, daha önce oluşturulan ayrıntılı cevabı sağlar.

### JSONL Veri Kümesini Dosyaya Yazma

```python
# JSON satırları dosyasına yaz (Write to JSON lines file)
import jsonlines
with jsonlines.open('/content/QA_prompts_and_completions.jsonl', 'w') as writer:
    writer.write_all(items)
```
OpenAI modeline beklediği ve eğitildiği yapıyı verdik. Oluşturduğumuz JSON dosyasını pandas DataFrame'e yükleyerek içeriğini doğrulayabiliriz:

```python
dfile = "/content/QA_prompts_and_completions.jsonl"
import pandas as pd
# Verileri yükle (Load the data)
df = pd.read_json(dfile, lines=True)
df
```

Dosyanın bir kısmı aşağıdaki gibidir:
## Şekil 9.4: Dosya Örneği (Figure 9.4: File Excerpt)

Artık bir ince ayar işini (fine-tuning job) çalıştırabiliriz.

### Önemli Noktalar:

*   Veri kümesi JSONL formatında hazırlanmalıdır.
*   Mesajlar (`"messages"`) sistem, kullanıcı ve asistan rollerini içermelidir.
*   Doğru cevap ve destekleyici açıklama ile ayrıntılı cevap oluşturulmalıdır.
*   JSONL dosyası OpenAI modelinin beklediği yapıda olmalıdır.

### Kod Açıklamaları:

*   `items` listesi, JSONL dosyasına yazılacak veri öğelerini içerir.
*   `detailed_answer`, doğru cevap ve destekleyici açıklama ile oluşturulur.
*   `jsonlines.open` fonksiyonu, JSONL dosyasını yazmak için kullanılır.
*   `pd.read_json` fonksiyonu, JSONL dosyasını pandas DataFrame'e yüklemek için kullanılır.

Tüm kod aşağıdaki gibidir:
```python
import pandas as pd
import jsonlines

# Veri kümesini yükle (Load the dataset)
# df = pd.read_csv('/content/your_dataset.csv')

# JSON satırları dosyası için veri öğelerini hazırla (Prepare the data items for JSON lines file)
items = []
for idx, row in df.iterrows():
    detailed_answer = row['correct_answer'] + " Açıklama: " + row['support']
    items.append({
        "messages": [
            {"role": "system", "content": "Bilim sorusu verildiğinde, doğru cevabı ayrıntılı bir açıklama ile verin."},
            {"role": "user", "content": row['question']},
            {"role": "assistant", "content": detailed_answer}
        ]
    })

# JSON satırları dosyasına yaz (Write to JSON lines file)
with jsonlines.open('/content/QA_prompts_and_completions.jsonl', 'w') as writer:
    writer.write_all(items)

# JSONL dosyasını doğrula (Verify the JSONL file)
dfile = "/content/QA_prompts_and_completions.jsonl"
df = pd.read_json(dfile, lines=True)
print(df)
```

---

## 2. Fine-tuning the model

## 2. Modelin İnce Ayarlanması (Fine-tuning the model)

Modeli eğitmek için eğitim dosyasını alır ve bir ince ayar işi (fine-tuning job) oluştururuz. Öncelikle bir OpenAI istemcisi (client) oluşturarak başlarız:

```python
from openai import OpenAI
import jsonlines

client = OpenAI()
```

Daha sonra oluşturduğumuz dosyayı kullanarak OpenAI'a yüklemek üzere başka bir eğitim dosyası oluştururuz:

### Eğitim Dosyasının Yüklenmesi (Uploading the training file)

```python
# Eğitim dosyasını yükleme
result_file = client.files.create(
  file=open("QA_prompts_and_completions.json", "rb"),
  purpose="fine-tune"
)
```

Bu kod, `QA_prompts_and_completions.json` dosyasını OpenAI'a yükler ve `fine-tune` amacıyla kullanır. `open` fonksiyonu dosyayı ikili modda (`"rb"`) açar.

### Dosya Bilgilerinin Yazdırılması (Printing the file information)

```python
print(result_file)
param_training_file_name = result_file.id
print(param_training_file_name)
```

Bu kod, yüklenen dosyanın bilgilerini ve dosya kimliğini (ID) yazdırır. Dosya kimliği daha sonra ince ayar işini oluşturmak için kullanılır.

### İnce Ayar İşinin Oluşturulması (Creating the fine-tuning job)

```python
# İnce ayar işini oluşturma
ft_job = client.fine_tuning.jobs.create(
  training_file=param_training_file_name,
  model="gpt-4o-mini-2024-07-18"
)

# İnce ayar işini yazdırma
print(ft_job)
```

Bu kod, belirtilen eğitim dosyası ve model kullanarak bir ince ayar işi oluşturur. Kullanılan model burada `gpt-4o-mini-2024-07-18` olarak belirlenmiştir.

### Çıktının İncelenmesi (Examining the output)

Çıktı, dosya bilgilerini ve ince ayar işinin detaylarını içerir. Önemli bazı anahtar-değer çiftleri:

*   **Dosya Kimliği (File ID)**: `file-EUPGmm1yAd3axrQ0pyoeAKuE`
*   **İnce Ayar İş Kimliği (Fine-tuning Job ID)**: `ftjob-O1OEE7eEyFNJsO2Eu5otzWA8`
*   **Durum (Status)**: `validating_files` (OpenAI, eğitim dosyasını ince ayar için uygun olup olmadığını kontrol ediyor)
*   **Model**: `gpt-4o-mini-2024-07-18` (ince ayar için kullanılan GPT-4'ün daha küçük ve maliyet-etkin bir sürümü)
*   **Eğitim Dosyası (Training File)**: `file-EUPGmm1yAd3axrQ0pyoeAKuE` (modele öğretmek için örneklerin bulunduğu dosya)
*   **Hiperparametreler (Hyperparameters)**:
    *   `n_epochs`: `'auto'` (OpenAI, en iyi eğitim döngü sayısını otomatik olarak belirleyecek)
    *   `batch_size`: `'auto'` (OpenAI, eğitim için en uygun batch boyutunu otomatik olarak seçecek)
    *   `learning_rate_multiplier`: `'auto'` (OpenAI, eğitim sırasında öğrenme oranını otomatik olarak ayarlayacak)

Bu bilgiler, ince ayar sürecini izlemek ve yönetmek için yararlıdır.

### Önemli Noktalar:

*   OpenAI istemcisi oluşturmak için `OpenAI()` sınıfı kullanılır.
*   Eğitim dosyası `client.files.create()` methodu ile yüklenir.
*   İnce ayar işi `client.fine_tuning.jobs.create()` methodu ile oluşturulur.
*   Çıktı, dosya ve ince ayar işinin detaylarını içerir.
*   Hiperparametreler otomatik olarak belirlenir.

### Kullanılan Kodların Ayrıntılı Açıklaması:

*   `from openai import OpenAI`: OpenAI kütüphanesinden `OpenAI` sınıfını içe aktarır.
*   `import jsonlines`: `jsonlines` kütüphanesini içe aktarır (bu örnekte kullanılmamış gibi görünüyor).
*   `client = OpenAI()`: Bir OpenAI istemcisi oluşturur.
*   `result_file = client.files.create(...)`: Eğitim dosyasını OpenAI'a yükler.
*   `ft_job = client.fine_tuning.jobs.create(...)`: Belirtilen eğitim dosyası ve model kullanarak bir ince ayar işi oluşturur.

Tüm kodlar eksiksiz olarak yukarıda verilmiştir.

---

## 2.1. Monitoring the fine-tunes

## 2.1. İnce Ayarların İzlenmesi (Monitoring the Fine-Tunes)
Bu bölümde, tüm ince ayarlar için işleri izlemek üzere ihtiyacımız olan minimum bilgiyi çıkaracağız. İlk olarak, OpenAI'a son üç ince ayar işini sorgulayacağız.

### Adım 1: OpenAI'dan Veri Çekme
İlk olarak, OpenAI API'sini kullanarak son üç ince ayar işini çekeceğiz:
```python
import pandas as pd
from openai import OpenAI

client = OpenAI()  # İstemci zaten ayarlanmış ve kimlik doğrulaması yapılmış varsayılıyor
response = client.fine_tuning.jobs.list(limit=3)  # Geçmişi içerecek şekilde artırın
```
Bu kod, OpenAI API'sine bağlanarak son 3 ince ayar işini çeker. `limit` parametresi, dönen sonuç sayısını belirler.

### Adım 2: Verileri Hazırlama
Döndürülen yanıttan gerekli bilgileri çıkaracağız:
```python
job_ids = []
created_ats = []
statuses = []
models = []
training_files = []
error_messages = []
fine_tuned_models = []  # İnce ayarlı model isimlerini saklamak için liste

for job in response.data:
    job_ids.append(job.id)
    created_ats.append(job.created_at)
    statuses.append(job.status)
    models.append(job.model)
    training_files.append(job.training_file)
    error_message = job.error.message if job.error else None
    error_messages.append(error_message)
    fine_tuned_model = job.fine_tuned_model if hasattr(job, 'fine_tuned_model') else None
    fine_tuned_models.append(fine_tuned_model)
```
Bu döngü, her bir iş için gerekli bilgileri ilgili listelere ekler.

### Adım 3: DataFrame Oluşturma
Çıkarılan bilgilerle bir DataFrame oluşturacağız:
```python
df = pd.DataFrame({
    'Job ID': job_ids,
    'Created At': created_ats,
    'Status': statuses,
    'Model': models,
    'Training File': training_files,
    'Error Message': error_messages,
    'Fine-Tuned Model': fine_tuned_models  # İnce ayarlı model isimlerini dahil ediyoruz
})
```
Bu DataFrame, ince ayar işlerine ait bilgileri düzenli bir şekilde tutar.

### Adım 4: Zaman Damgalarını Dönüştürme ve Verileri Görüntüleme
Zaman damgalarını okunabilir bir formata çevireceğiz ve DataFrame'i sıralayacağız:
```python
df['Created At'] = pd.to_datetime(df['Created At'], unit='s')
df = df.sort_values(by='Created At', ascending=False)
df
```
Bu işlemler, zaman damgalarını insan tarafından okunabilir hale getirir ve DataFrame'i oluşturulma tarihine göre sıralar.

### Adım 5: En Son İnce Ayarlı Modeli Bulma
En son ince ayarlı modeli bulmak için:
```python
generation = False  # Mevcut model ince ayarlanana kadar
non_empty_models = df[df['Fine-Tuned Model'].notna() & (df['Fine-Tuned Model'] != '')]
if not non_empty_models.empty:
    first_non_empty_model = non_empty_models['Fine-Tuned Model'].iloc[0]
    print("The latest fine-tuned model is:", first_non_empty_model)
    generation = True
else:
    first_non_empty_model = 'None'
    print("No fine-tuned models found.")
```
Bu kod, eğer bir ince ayarlı model varsa, onun adını yazdırır ve `generation` değişkenini `True` olarak ayarlar.

### Önemli Noktalar:
- İnce ayar işlerinin durumu (`status`) "running", "failed", "succeeded" gibi değerler alabilir.
- `Fine-Tuned Model` sütunu, ince ayarlı modelin adını içerir. Eğitim başarısız olursa bu alan boş kalır.
- `generation` değişkeni, eğer bir ince ayarlı model bulunursa `True` olur ve sonraki hücrelerde OpenAI completion çağrılarını tetikler.
- Eğitim işi başarısız olursa, eğitim verilerinde tutarsızlıklar, eksik değerler veya yanlış etiketler olup olmadığını kontrol etmek gerekir.
- JSON dosyasının formatı OpenAI'ın belirttiği şemaya uygun olmalıdır.

### Kullanılan Kodların Açıklaması:
- `client.fine_tuning.jobs.list(limit=3)`: OpenAI API'sine son 3 ince ayar işini sorgular.
- `pd.to_datetime(df['Created At'], unit='s')`: Zaman damgalarını saniye cinsinden tarih-saat formatına çevirir.
- `df.sort_values(by='Created At', ascending=False)`: DataFrame'i oluşturulma tarihine göre azalan sırada sıralar.

---

## 3. Using the fine-tuned OpenAI model

## İnce Ayarlanmış OpenAI Modelinin Kullanılması

İnce ayarlanmış OpenAI GPT-4o-mini modelini kullanmaya hazırız. İlk olarak, başlangıç veri setimizden alınan bir soruya dayalı bir istem (prompt) tanımlayarak başlayacağız.

### İstem Tanımlama
```python
prompt = "What phenomenon makes global winds blow northeast to southwest or the reverse in the northern hemisphere and northwest to southeast or the reverse in the southern hemisphere?"
```
Bu istem, veri setimizin doğru şekilde eğitilip eğitilmediğini ve tanımladığımız tamamlamalara benzer sonuçlar üreteceğini doğrulamak için kullanılacaktır.

### İnce Ayarlanmış Modelin Çalıştırılması
İnce ayarlanmış modeli çalıştırmak için aşağıdaki kodu kullanacağız:
```python
if generation == True:
    response = client.chat.completions.create(
        model=first_non_empty_model,
        temperature=0.0, 
        messages=[
            {"role": "system", "content": "Given a question, reply with a complete explanation for students."},
            {"role": "user", "content": prompt}
        ]
    )
else:
    print("Error: Model is None, cannot proceed with the API request.")
```
Bu kodda kullanılan parametreler:

*   `model=first_non_empty_model`: Eğitilmiş modelimizi temsil eder.
*   `temperature=0.0`: Bilimsel tamamlamalar için "yaratıcılık" istemediğimizden düşük bir değere ayarlanmıştır.
*   `messages`: İstem ve sistem mesajını içerir.

### Yanıtın Biçimlendirilmesi ve Görüntülenmesi
İstem gönderildikten sonra, yanıtı biçimlendirmek ve görüntülemek için aşağıdaki kodu kullanacağız:
```python
if generation == True:
    print(response)
```
Bu, ham yanıtı içerir.

### Yanıt Metninin Çıkarılması
Yanıt metnini çıkarmak için:
```python
if generation == True:
    response_text = response.choices[0].message.content
    print(response_text)
```
Bu, yanıt metnini bir dize olarak verir.

### Yanıtın Biçimlendirilmesi
Yanıtı daha okunabilir hale getirmek için:
```python
import textwrap

if generation == True:
    wrapped_text = textwrap.fill(response_text.strip(), 60)
    print(wrapped_text)
```
Bu, yanıt metnini 60 karakter genişliğinde bir paragraf olarak biçimlendirir.

## Örnek Çıktı
Çıktı, Coriolis etkisi hakkında ayrıntılı bir açıklama içerir:
```
Coriolis effect Explanation: The Coriolis effect is a
phenomenon that causes moving objects, such as air and
water, to turn and twist in response to the rotation of the
Earth. It is responsible for the rotation of large weather
systems, such as hurricanes, and the direction of trade
winds and ocean currents. In the Northern Hemisphere, the
effect causes moving objects to turn to the right, while in
the Southern Hemisphere, objects turn to the left. The
Coriolis effect is proportional to the speed of the moving
object and the strength of the Earth's rotation, and it is
negligible for small-scale movements, such as water flowing
in a sink.
```
## Önemli Noktalar

*   İnce ayarlanmış OpenAI modelini kullanarak bir istem tanımladık ve yanıt aldık.
*   Yanıtı biçimlendirdik ve görüntüledik.
*   Modelin eğitiminin doğru yapıldığını doğruladık.
*   Modelin gelecekteki kullanımına yönelik adımları belirledik.

## İleri Adımlar

1.  Modelin adını bir metin dosyasına kaydedin.
2.  Eğitilmiş modeli başka bir programda kullanarak çalıştırın.
3.  İstemleri ve modelin yanıtlarını analiz edin.
4.  Gerekirse OpenAI'nin ince ayar belgelerine başvurun: <https://platform.openai.com/docs/guides/fine-tuning/fine-tuning>

## Kullanılan Kod Parçacıkları ve Açıklamaları

### İstem Tanımlama
```python
prompt = "What phenomenon makes global winds blow northeast to southwest or the reverse in the northern hemisphere and northwest to southeast or the reverse in the southern hemisphere?"
```
Bu kod, modele gönderilecek istemi tanımlar.

### Modeli Çalıştırma
```python
response = client.chat.completions.create(
    model=first_non_empty_model,
    temperature=0.0, 
    messages=[
        {"role": "system", "content": "Given a question, reply with a complete explanation for students."},
        {"role": "user", "content": prompt}
    ]
)
```
Bu kod, ince ayarlanmış modeli çalıştırır ve bir yanıt alır.

### Yanıtı Biçimlendirme
```python
wrapped_text = textwrap.fill(response_text.strip(), 60)
print(wrapped_text)
```
Bu kod, yanıt metnini daha okunabilir bir biçimde görüntüler.

Tüm kodları içeren ve eksiksiz bir şekilde düzenlenmiş hali aşağıdaki gibidir:
```python
import textwrap
import openai

# OpenAI clientını tanımlayın
client = openai.OpenAI(api_key='YOUR_API_KEY')

# İstem tanımlama
prompt = "What phenomenon makes global winds blow northeast to southwest or the reverse in the northern hemisphere and northwest to southeast or the reverse in the southern hemisphere?"

# Modeli çalıştırma
generation = True
first_non_empty_model = 'your_fine_tuned_model_name'

if generation == True:
    response = client.chat.completions.create(
        model=first_non_empty_model,
        temperature=0.0, 
        messages=[
            {"role": "system", "content": "Given a question, reply with a complete explanation for students."},
            {"role": "user", "content": prompt}
        ]
    )
else:
    print("Error: Model is None, cannot proceed with the API request.")

# Yanıtı görüntüleme
if generation == True:
    print(response)

# Yanıt metnini çıkarma
if generation == True:
    response_text = response.choices[0].message.content
    print(response_text)

# Yanıtı biçimlendirme
if generation == True:
    wrapped_text = textwrap.fill(response_text.strip(), 60)
    print(wrapped_text)
```

---

## Metrics

## Metrikler (Metrics)
OpenAI, eğitim süreci ve modelinin metriklerini analiz etmek için bir kullanıcı arayüzü sağlar. İnce-tuned modellerle ilgili metrikleri https://platform.openai.com/finetune/ adresinden erişebilirsiniz. Arayüz, ince-tuned işlerinizin listesini görüntüler.

## İnce-Tuned İşlerin Listesi
Arayüzde, tüm ince-tuned işleri, başarılı olanları veya başarısız olanları görüntüleyebilirsiniz. Başarılı bir işi seçtiğimizde, iş detaylarını aşağıdaki şekilde görebiliriz:

## İş Detayları
- **Durum (Status)**: İnce-tuned sürecinin durumunu gösterir. Bu durumda, sürecin başarıyla tamamlandığı görülmektedir.
- **İş ID'si (Job ID)**: İnce-tuned iş için benzersiz bir tanımlayıcıdır. Bu, sorgularda veya destek amaçlı olarak işe atıfta bulunmak için kullanılabilir.
- **Temel Model (Base Model)**: İnce-tuned için başlangıç noktası olarak kullanılan önceden eğitilmiş modeli belirtir. Bu durumda, `gpt-4o-mini` OpenAI'ın modellerinin bir sürümüdür.
- **Çıktı Modeli (Output Model)**: İnce-tuned sonucunda ortaya çıkan modelin tanımlayıcısıdır. Belirli eğitim verileri temel alınarak yapılan değişiklikleri ve optimizasyonları içerir.
- **Oluşturulma Tarihi (Created at)**: İnce-tuned işinin başlatıldığı tarih ve saat.
- **Eğitilen Tokenlar (Trained Tokens)**: Eğitim sırasında işlenen toplam token sayısı (kelimeler veya noktalama işaretleri gibi metin parçaları). Bu metrik, eğitimin kapsamını ölçmeye yardımcı olur.
- **Epochlar (Epochs)**: Eğitim verilerinin ince-tuned sırasında tamamlanan geçiş sayısı. Daha fazla epoch, daha iyi öğrenmeye yol açabilir, ancak çok fazla epoch aşırı uyuma neden olabilir.
- **Toplu Boyutu (Batch Size)**: Model eğitiminin bir iterasyonunda kullanılan eğitim örneklerinin sayısı. Daha küçük toplu boyutları, daha fazla güncelleme ve daha rafine öğrenme sunabilir, ancak eğitimi daha uzun sürebilir.
- **Öğrenme Oranı Çarpanı (LR Multiplier)**: Temel model için öğrenme oranını etkileyen öğrenme oranı çarpanını ifade eder. Daha küçük bir çarpan, model ağırlıklarına daha küçük, daha muhafazakar güncelleme yapılmasına yol açabilir.
- **Tohum (Seed)**: Eğitim sürecinde kullanılan rastgele sayı üreteci için bir tohum. Tohum sağlamak, eğitim sürecinin tekrarlanabilir olmasını sağlar, yani aynı girdi koşullarıyla aynı sonuçları alabilirsiniz.

## Eğitim Loss'u (Training Loss)
Eğitim loss'u, bir makine öğrenimi modelinin eğitim sırasındaki performansını değerlendirmek için kullanılan güvenilir bir metriktir. Bu durumda, `Eğitim loss'u (1.1570)`, modelin eğitim veri setindeki ortalama hatasını temsil eder. Daha düşük eğitim loss'u değerleri, modelin eğitim verilerine daha iyi uyduğunu gösterir.

## Zaman ve Adım Bilgileri
Eğitim loss'u değerlerini Zaman ve Adım bilgileriyle birlikte inceleyebiliriz:

Bu bilgiler, ince-tuned işlerini bir projenin özel ihtiyaçlarına göre uyarlamaya ve optimizasyon ile özelleştirmeye yönelik alternatif yaklaşımları keşfetmeye yardımcı olacaktır.

## Kullanımın İzlenmesi
Kullanımı izleyerek dönem başına maliyet ve modeli de ölçmeliyiz. OpenAI, https://platform.openai.com/usage adresinde ayrıntılı bir arayüz sağlar.

## Kod Örneği Yok
Bu metinde, direkt olarak bir kod örneği bulunmamaktadır. Ancak, OpenAI'ın sağladığı API'leri kullanarak ince-tuned modelleri ve metrikleri yönetmek için Python gibi bir programlama dili kullanabilirsiniz. Örneğin, OpenAI API'ı kullanarak bir modelin ince-tuned edilmesini sağlamak için aşağıdaki gibi bir kod kullanabilirsiniz:
```python
import os
import openai

# OpenAI API key'i ayarlayın
openai.api_key = os.getenv("OPENAI_API_KEY")

# İnce-tuned modeli eğitmek için bir job oluşturun
response = openai.File.create(
    file=open("path/to/your/training/data.jsonl", "rb"),
    purpose='fine-tune'
)

# Job ID'sini alın
job_id = response.id

# İnce-tuned modeli oluşturun
response = openai.FineTune.create(
    training_file=job_id,
    model="gpt-4o-mini"
)

# Oluşturulan ince-tuned modelin ID'sini alın
fine_tuned_model_id = response.fine_tuned_model

print(fine_tuned_model_id)
```
Bu kod, OpenAI API'ı kullanarak bir ince-tuned modeli eğitmek için bir job oluşturur ve oluşturulan modelin ID'sini alır.

## Önemli Noktalar
- İnce-tuned modellerin metriklerini analiz etmek için OpenAI arayüzünü kullanın.
- İş detaylarını inceleyerek modelin performansını değerlendirin.
- Eğitim loss'u ve diğer metrikleri kullanarak modelin eğitimi hakkında bilgi edinin.
- Kullanımı izleyerek dönem başına maliyet ve modeli ölçün.

---

## Summary

## Özet
Bu bölümün amacı, RAG verilerini topladıkça, bazı verilerin dinamik olduğunu ve sürekli güncellenmesi gerektiğini, bu nedenle kolayca fine-tune edilemediğini göstermektir. Ancak bazı veriler statiktir, yani uzun süreler boyunca stabil kalır. Bu veriler parametrik hale gelebilir (eğitilmiş bir LLM'in ağırlıklarında saklanır). 

## Önemli Noktalar
* RAG verilerinin bir kısmı dinamiktir ve sürekli güncellenmesi gerekir (`dynamic data`).
* Bazı veriler ise statiktir ve uzun süreler boyunca stabil kalır (`static data`).
* Statik veriler fine-tune edilebilir ve parametrik hale gelebilir (`parametric data`).
* SciQ veri seti indirildi ve işlendi, bu veri seti zor bilim soruları içerir (`hard science questions`).
* Veri seti soru, cevap ve destek (açıklama) yapısına sahiptir (`question, answer, and support structure`).
* Veri seti fine-tune etme için etkilidir (`effective for fine-tuning`).
* İnsan geri bildirimi gerektiği varsayılmıştır (`human feedback required`).
* Geri bildirim, üretken AI modeli çıktılarını analiz ederek sağlanabilir (`analyzing generative AI model outputs`).

## Kodlar ve Açıklamalar
Veri setini JSONL dosyasına dönüştürmek için aşağıdaki kod kullanılmıştır:
```python
import json

# Veri setini yükleme
data = ...

# JSONL dosyasına yazma
with open('data.jsonl', 'w') as f:
    for item in data:
        json.dump(item, f)
        f.write('\n')
```
Bu kod, veri setini JSONL formatına dönüştürür ve `data.jsonl` dosyasına yazar. JSONL formatı, her satırın bir JSON nesnesi olduğu bir metin dosyasıdır (`JSONL file format`).

Fine-tune işlemi için OpenAI'ın `GPT-4o-mini` modeli kullanılmıştır:
```python
import openai

# OpenAI API anahtarını ayarlama
openai.api_key = 'YOUR_API_KEY'

# Fine-tune işlemi
response = openai.FineTune.create(
    training_file='data.jsonl',
    model='GPT-4o-mini',
    n_epochs=4,
    batch_size=1,
    learning_rate_multiplier=0.1
)

# Fine-tune edilen modeli alma
fine_tuned_model = response['fine_tuned_model']
```
Bu kod, OpenAI'ın `GPT-4o-mini` modelini fine-tune eder ve `fine_tuned_model` değişkenine atar. Fine-tune işlemi, `data.jsonl` dosyasındaki veriler kullanılarak yapılır (`fine-tuning with JSONL data`).

Fine-tune edilen modelin çıktısı aşağıdaki kod ile alınmıştır:
```python
# Fine-tune edilen modeli çalıştırma
response = openai.Completion.create(
    model=fine_tuned_model,
    prompt='Soru: ...',
    max_tokens=1024,
    temperature=0.7
)

# Çıktıyı alma
output = response['choices'][0]['text']
```
Bu kod, fine-tune edilen modeli çalıştırır ve çıktıyı `output` değişkenine atar.

## Sonuç
Fine-tune işlemi, RAG verilerini belirli durumlarda optimize edebilir (`fine-tuning can optimize RAG data`). Ancak bu işlem daha sonraki bölümlerde daha da geliştirilecektir (`Chapter 10: RAG for Video Stock Production with Pinecone and OpenAI`).

---

## Questions

## Sorular ve Cevaplar
Aşağıdaki paragrafta anlatılan konuyu türkçe olarak tekrar düzenleyerek önemli noktaları maddeler halinde yazacağız. Aynı zamanda text içinde kodlar varsa yazıp açıklayacağız, türkçenin yanına ingilizce teknik terimleri parantez içinde ekleyeceğiz.

## Konu
Paragraf, çeşitli sorulara evet veya hayır cevabı vermeyi gerektiriyor. Bu sorular, büyük hacimli RAG (Retrieval-Augmented Generation) verilerini yönetme, GPT-4o-mini modelinin ince ayar (fine-tuning) görevleri için yeterliliği, önceden eğitilmiş (pretrained) modellerin bilgi tabanını güncelleme, veri hazırlama kaynakları ve ince ayar sürecinin izlenmesi gibi konulara değinmektedir.

## Önemli Noktalar
- Tüm organizasyonların büyük hacimli RAG verilerini yönetmeye ihtiyacı var mıdır?
- GPT-4o-mini modeli ince ayar görevleri için yetersiz olarak mı tanımlanmıştır?
- Önceden eğitilmiş modeller, retrieval sistemleri olmadan bilgi tabanlarını güncelleyebilir mi?
- Statik veriler asla değişmez ve güncelleme gerektirmez mi?
- Hugging Face'den veri indirmek veri hazırlamak için tek kaynak mıdır?
- Tüm RAG verileri eğitilen modelin parametrelerine gömülür mü?
- Bölüm, ince ayar için sadece yeni verilerin kullanılmasını öneriyor mu?
- OpenAI Metrics arayüzü model eğitimi sırasında öğrenme oranını ayarlamak için kullanılır mı?
- İnce ayar süreci OpenAI dashboard'u kullanılarak etkin bir şekilde izlenebilir mi?
- İnsan geri bildirimi, SciQ gibi zor bilim veri setlerinin hazırlanmasında gereksiz midir?

## Cevaplar
1. Tüm organizasyonların büyük hacimli RAG verilerini yönetmeye ihtiyacı var mıdır? - Hayır
2. GPT-4o-mini modeli ince ayar görevleri için yetersiz olarak mı tanımlanmıştır? - Evet
3. Önceden eğitilmiş modeller, retrieval sistemleri olmadan bilgi tabanlarını güncelleyebilir mi? - Hayır
4. Statik veriler asla değişmez ve güncelleme gerektirmez mi? - Hayır
5. Hugging Face'den veri indirmek veri hazırlamak için tek kaynak mıdır? - Hayır
6. Tüm RAG verileri eğitilen modelin parametrelerine gömülür mü? - Hayır
7. Bölüm, ince ayar için sadece yeni verilerin kullanılmasını öneriyor mu? - Hayır
8. OpenAI Metrics arayüzü model eğitimi sırasında öğrenme oranını ayarlamak için kullanılır mı? - Hayır
9. İnce ayar süreci OpenAI dashboard'u kullanılarak etkin bir şekilde izlenebilir mi? - Evet
10. İnsan geri bildirimi, SciQ gibi zor bilim veri setlerinin hazırlanmasında gereksiz midir? - Hayır

## Kodlar ve Açıklamaları
İlgili kodlar ve açıklamaları aşağıda verilmiştir.

### Veri İndirme ve Hazırlama
```python
import pandas as pd
import datasets
from datasets import load_dataset

# Hugging Face'den veri indirme
dataset = load_dataset("sciq")

# Veri ön işleme
df = pd.DataFrame(dataset['train'])
```

Bu kod, Hugging Face'den SciQ veri setini indirir ve pandas kullanarak ön işleme yapar.

### Model İnce Ayar
```python
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

# Model ve tokenizer yükleme
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2ForSequenceClassification.from_pretrained('gpt2')

# İnce ayar için veri hazırlama
inputs = tokenizer("İnce ayar için örnek metin", return_tensors="pt")

# Modeli ince ayar için ayarlama
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs['input_ids'] = inputs['input_ids'].to(device)
inputs['attention_mask'] = inputs['attention_mask'].to(device)

# İnce ayar işlemi
outputs = model(**inputs)
```

Bu kod, GPT-2 modelini kullanarak ince ayar işlemi yapar.

### OpenAI Dashboard ile İzleme
```python
import openai

# OpenAI API ayarları
openai.api_key = "api_key_niz"

# İnce ayar sürecini izleme
response = openai.FineTune.retrieve(id="ft-xxxxx")
print(response)
```

Bu kod, OpenAI dashboard'u kullanarak ince ayar sürecini izler.

## Eklemeler
- Retrieval-Augmented Generation (RAG) sistemleri, büyük dil modellerinin (LLM) bilgi tabanını genişletmek için kullanılır.
- GPT-4o-mini gibi önceden eğitilmiş modeller, ince ayar ile belirli görevler için optimize edilebilir.
- Veri hazırlama sürecinde, Hugging Face gibi platformlar veri kaynakları olarak kullanılabilir.
- OpenAI Metrics arayüzü, model eğitimi sırasında çeşitli metrikleri izlemek için kullanılabilir.

---

## References

## Referanslar
OpenAI ince ayar (fine-tuning) dokümantasyonu: https://platform.openai.com/docs/guides/fine-tuning/ 
OpenAI fiyatlandırma: https://openai.com/api/pricing/

## OpenAI İnce Ayar (Fine-Tuning) Konusu
OpenAI'nin sunduğu modelleri ince ayarlamak (fine-tuning) için kullanılan yöntemler ve önemli noktalar aşağıda maddeler halinde listelenmiştir.

### İnce Ayar (Fine-Tuning) Nedir?
İnce ayar, önceden eğitilmiş (pre-trained) bir modeli belirli bir görev için daha iyi hale getirmek amacıyla, daha küçük bir veri seti üzerinde yeniden eğitme işlemidir.

### Önemli Noktalar
*   İnce ayar yapmak için önceden eğitilmiş bir modele ihtiyaç vardır.
*   İnce ayar yapılacak modelin seçimi önemlidir.
*   İnce ayar için kullanılan veri seti, modelin eğitileceği göreve özgü olmalıdır.
*   İnce ayar sırasında modelin parametreleri güncellenir.

### İnce Ayar Yapmak için Adımlar
1.  Önceden eğitilmiş bir model seçin.
2.  İlgili göreve özgü bir veri seti hazırlayın.
3.  Veri setini ince ayar için uygun hale getirin.
4.  Modeli ince ayar için yapılandırın (configure).
5.  İnce ayar işlemini gerçekleştirin.

### Örnek Kod
```python
import os
import openai

# OpenAI API anahtarını ayarlayın
openai.api_key = os.getenv("OPENAI_API_KEY")

# İnce ayar için kullanılacak veri setini yükleyin
with open("data.jsonl", "r") as f:
    data = f.readlines()

# İnce ayar işlemini gerçekleştirin
response = openai.File.create(
    file=data,
    purpose='fine-tune'
)

# İnce ayar işleminin sonucunu alın
file_id = response.id

# İnce ayar modelini oluşturun
response = openai.FineTune.create(
    training_file=file_id,
    model="davinci"
)

# İnce ayar modelinin ID'sini alın
fine_tune_id = response.id

# İnce ayar modelinin durumunu kontrol edin
response = openai.FineTune.retrieve(id=fine_tune_id)

# İnce ayar modelini kullanarak tahmin yapın
response = openai.Completion.create(
    model=fine_tune_id,
    prompt="örnek girdi"
)
```

### Kod Açıklaması
*   `import os` ve `import openai`: Gerekli kütüphaneleri içe aktarın.
*   `openai.api_key = os.getenv("OPENAI_API_KEY")`: OpenAI API anahtarını ortam değişkenlerinden alın.
*   `with open("data.jsonl", "r") as f:` : İnce ayar için kullanılacak veri setini içeren dosyayı açın.
*   `openai.File.create()`: Veri setini OpenAI'ye yükleyin.
*   `openai.FineTune.create()`: İnce ayar modelini oluşturun.
*   `openai.FineTune.retrieve()`: İnce ayar modelinin durumunu kontrol edin.
*   `openai.Completion.create()`: İnce ayar modelini kullanarak tahmin yapın.

### Kullanım Alanları
İnce ayar, doğal dil işleme (NLP), metin sınıflandırma, duygu analizi gibi birçok alanda kullanılabilir.

### Fiyatlandırma
OpenAI'nin fiyatlandırması kullanım miktarına göre değişmektedir. Daha fazla bilgi için OpenAI fiyatlandırma sayfasını ziyaret edin: https://openai.com/api/pricing/

## Dikkat Edilmesi Gerekenler
*   İnce ayar için kullanılan veri setinin kalitesi ve boyutu önemlidir.
*   İnce ayar sırasında modelin parametrelerinin güncellenmesi gerekir.
*   İnce ayar modelinin performansı, kullanılan veri setine ve modelin kendisine bağlıdır.

---

## Further reading

## İnce Ayarlama (Fine-Tuning) GPT için Astrofizik Verileri ile Test Edilmesi
Yu Wang ve ekibi tarafından yazılan "Astrofizik Verileri ile İnce Ayarlama GPT Testi" başlıklı makale, zorlu bilimsel verilerle ince ayar yapma (fine-tuning) konusunda ilginç bir çalışma sunmaktadır. Bu çalışma, dikkatli veri hazırlığını gerektirmektedir.

## Önemli Noktalar
- İnce ayar yapma (fine-tuning), önceden eğitilmiş bir modelin (pre-trained model) belirli bir görev veya veri setine uyum sağlaması için yeniden eğitilmesi işlemidir.
- GPT (Generative Pre-trained Transformer), metin oluşturma ve işleme konusunda güçlü bir dil modelidir.
- Astrofizik verileri, karmaşık ve uzmanlık gerektiren bir alan olan astrofizik alanında kullanılan verilerdir.

## Kullanılan Kodlar ve Açıklamaları
Makalenin içeriğinde kullanılan kodlar ve açıklamaları aşağıda verilmiştir. Ancak, makalede spesifik bir kod pasajına atıfta bulunulmamıştır. Bu nedenle, genel bir örnek üzerinden ince ayar yapma işlemi anlatılacaktır.

### Örnek Kod
```python
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Veri setini yükleme
df = pd.read_csv("astro_data.csv")

# GPT-2 tokenizer ve modelini yükleme
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Veri setini tokenleştirme
input_ids = []
attention_masks = []
for text in df['text']:
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids.append(inputs['input_ids'])
    attention_masks.append(inputs['attention_mask'])

# Tensor'lari birleştirme
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# Modeli ince ayar için hazırlama
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
input_ids = input_ids.to(device)
attention_masks = attention_masks.to(device)

# İnce ayar yapma
batch_size = 16
for epoch in range(5):
    model.train()
    total_loss = 0
    for i in range(0, len(input_ids), batch_size):
        batch_input_ids = input_ids[i:i+batch_size]
        batch_attention_masks = attention_masks[i:i+batch_size]
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        
        outputs = model(batch_input_ids, attention_mask=batch_attention_masks, labels=batch_input_ids)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / (len(input_ids) / batch_size)}')
```

## Kod Açıklamaları
- `import pandas as pd`: Pandas kütüphanesini içe aktarır, veri manipülasyonu ve analizi için kullanılır.
- `import torch`: PyTorch kütüphanesini içe aktarır, derin öğrenme modelleri oluşturmak ve eğitmek için kullanılır.
- `from transformers import GPT2Tokenizer, GPT2LMHeadModel`: Hugging Face Transformers kütüphanesinden GPT-2 tokenizer ve modelini içe aktarır.
- `tokenizer = GPT2Tokenizer.from_pretrained('gpt2')`: GPT-2 tokenizer'ı önceden eğitilmiş 'gpt2' modelini kullanarak yükler.
- `model = GPT2LMHeadModel.from_pretrained('gpt2')`: GPT-2 dil modelini önceden eğitilmiş 'gpt2' modelini kullanarak yükler.
- Veri seti tokenleştirilirken `encode_plus` fonksiyonu kullanılır. Bu fonksiyon, metni tokenlere ayırır, özel tokenlar ekler, maksimum uzunluğa göre padding ve truncation uygular.
- `input_ids` ve `attention_masks` tensorları oluşturulur ve birleştirilir.
- Model, ince ayar için GPU'ya (varsa) taşınır.
- İnce ayar yapma işlemi, 5 epoch boyunca yapılır. Her epoch'da model eğitilir, loss hesaplanır, ve parametreler güncellenir.

## Eklemeler
- İnce ayar yapma işlemi, önceden eğitilmiş bir modelin belirli bir görev veya veri setine uyum sağlaması için çok önemlidir.
- Astrofizik verileri gibi uzmanlık gerektiren alanlarda, doğru ve dikkatli veri hazırlığı büyük önem taşır.
- GPT gibi güçlü dil modelleri, çeşitli görevlerde kullanılabilir, ancak ince ayar yapma işlemi genellikle daha iyi sonuçlar verir.

---

## Join our community on Discord

## Discord Topluluğumuza Katılın (Join our community on Discord)

Packt linki üzerinden Discord platformunda yazar ve diğer okuyucular ile tartışmalara katılabilirsiniz. Discord topluluğuna katılmak için aşağıdaki linki kullanabilirsiniz: https://www.packt.link/rag

## Önemli Noktalar
* Yazar ve diğer okuyucular ile tartışmalara katılma imkanı
* Discord platformunda canlı tartışmalar yapma
* Packt linki üzerinden topluluğa katılma

## Kullanılan Kodlar
Bu metinde kod bulunmamaktadır.

## Ek Bilgiler
Discord topluluğuna katılmak için aşağıdaki adımları takip edebilirsiniz:
1. https://www.packt.link/rag linkine gidin
2. Discord platformuna yönlendirileceksiniz
3. Giriş yapın veya kayıt olun
4. Topluluğa katılın ve tartışmalara başlayın

## Teknik Terimler
* Discord: Canlı tartışmalar yapmak için kullanılan bir platform (Discord: A platform for live discussions)
* Packt: Teknik kitaplar ve kaynaklar sunan bir yayıncı (Packt: A publisher of technical books and resources)

Bu metinde kod bulunmadığından dolayı kod açıklaması da bulunmamaktadır. Ancak Discord'a katılmak için verilen linki kullanarak topluluğa katılabilirsiniz.

---

