## Why Retrieval Augmented Generation?

## Retrieval Augmented Generation (RAG) Nedir?
Retrieval Augmented Generation (RAG), üretken yapay zeka (Generative AI) modellerinin sınırlamalarını aşmak için kullanılan bir çerçevedir (framework). Üretken yapay zeka modelleri, yalnızca eğitildikleri verilere dayanarak cevaplar üretebilirler ve bu verilerin dışında kalan bilgilere ilişkin sorulara doğru cevaplar veremezler. Bu durum, hatalı veya uygunsuz çıktılarla sonuçlanabilir.

## RAG'ın Avantajları
RAG'ın temel avantajları şunlardır:
* **Esneklik**: RAG, metin, resim veya ses gibi her türlü veriye uygulanabilir.
* **Doğruluk**: RAG, harici kaynaklardan alınan verilerle daha doğru ve bağlamı ilgili cevaplar üretebilir.
* **Güç**: RAG ile entegre üretken yapay zeka modelleri, olağanüstü verimlilik ve güç sağlar.

## RAG Konfigürasyonları
RAG'ın üç ana konfigürasyonu vardır:
* **Naïve RAG**: Basit anahtar kelime arama ve eşleştirme tabanlı RAG'dir.
* **Advanced RAG**: Vektör arama ve indeks tabanlı RAG'dir.
* **Modular RAG**: Hem naïve hem de advanced RAG'ı dikkate alan modüler bir RAG'dir.

## RAG ve Fine-Tuning
RAG ve fine-tuning, üretken yapay zeka modellerini geliştirmek için kullanılan iki farklı yaklaşımdır. RAG, harici kaynaklardan alınan verilerle daha doğru cevaplar üretebilirken, fine-tuning, modelin eğitildiği verilere dayanarak daha doğru cevaplar üretebilir.

## RAG Ekosistemi
RAG, yalnızca bir ekosistem içinde var olabilir. Veriler bir yerden gelmeli ve işlenmelidir. Retrieval, verileri almak için organize bir ortam gerektirir ve üretken yapay zeka modellerinin girdi kısıtlamaları vardır.

## Python ile RAG Uygulaması
Bu bölümde, Python programlama dili kullanılarak naïve RAG, advanced RAG ve modular RAG uygulamaları yapılacaktır.

### Naïve RAG Uygulaması
Naïve RAG uygulaması için aşağıdaki kod kullanılacaktır:
```python
import pandas as pd

# Veri setini yükleme
df = pd.read_csv('data.csv')

# Anahtar kelime arama ve eşleştirme
def naive_rag(query):
    results = df[df['text'].str.contains(query)]
    return results

# Sorgu yapma
query = 'arama kelimesi'
results = naive_rag(query)
print(results)
```
Bu kod, `data.csv` adlı veri setini yükler ve anahtar kelime arama ve eşleştirme yaparak sonuçları döndürür.

### Advanced RAG Uygulaması
Advanced RAG uygulaması için aşağıdaki kod kullanılacaktır:
```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Veri setini yükleme
df = pd.read_csv('data.csv')

# Vektör arama ve indeks tabanlı retrieval
def advanced_rag(query):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    query_embedding = model.encode(query)
    doc_embeddings = model.encode(df['text'])
    similarities = cosine_similarity([query_embedding], doc_embeddings).flatten()
    results = df.iloc[np.argsort(-similarities)]
    return results

# Sorgu yapma
query = 'arama kelimesi'
results = advanced_rag(query)
print(results)
```
Bu kod, `data.csv` adlı veri setini yükler ve vektör arama ve indeks tabanlı retrieval yaparak sonuçları döndürür.

### Modular RAG Uygulaması
Modular RAG uygulaması için aşağıdaki kod kullanılacaktır:
```python
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Veri setini yükleme
df = pd.read_csv('data.csv')

# Naïve RAG ve advanced RAG'ı birleştirme
def modular_rag(query):
    naive_results = naive_rag(query)
    advanced_results = advanced_rag(query)
    results = pd.concat([naive_results, advanced_results])
    return results

# Sorgu yapma
query = 'arama kelimesi'
results = modular_rag(query)
print(results)
```
Bu kod, `data.csv` adlı veri setini yükler ve naïve RAG ile advanced RAG'ı birleştirerek sonuçları döndürür.

---

## What is RAG?

## RAG Nedir?
RAG (Retrieval-Augmented Generation), bir Büyük Dil Modeli (LLM - Large Language Model) tarafından doğru bir şekilde cevap verilemeyen durumlarda, modele eksik olan bilgiyi sağlayarak daha doğru çıktılar üretmesini sağlayan bir çerçevedir (Lewis et al., 2020). RAG, optimizasyonlu bilgi erişimi görevlerini gerçekleştirir ve oluşturulan ekosistem, bu bilgiyi girdi (kullanıcı sorgusu veya otomatik prompt) ile birleştirerek geliştirilmiş çıktı üretir.

## RAG'ın Temel Bileşenleri
RAG'ın iki ana bileşeni vardır:
- **Bilgi Erişimi (Retrieval)**: Kullanıcı sorgusuna veya otomatik prompta ilgili bilgilerin bulunması.
- **Oluşturma (Generation)**: Erişilen bilgilerin kullanılarak çıktı üretilmesi.

## RAG'ın Çalışma Prensibi
RAG, bir kütüphanede araştırma yapan bir öğrenci gibi düşünülebilir. Öğrenci, RAG hakkında bir makale yazmak istiyor ve gerekli bilgiye sahip olmak için kütüphaneye başvuruyor. 
1. **Bilgi Erişimi**: Öğrenci, RAG ile ilgili kitapları kütüphanede arar ve gerekli bilgiyi toplar.
2. **Oluşturma**: Öğrenci, topladığı bilgiyi kullanarak makalesini yazar.

## RAG'ın Önemi
RAG, Büyük Dil Modelleri'nin (LLM) sınırlamalarını aşmak için tasarlanmıştır. LLM'ler, geniş bir bilgi yelpazesinde eğitilmiş olmalarına rağmen, her zaman doğru veya güncel bilgiye sahip olmayabilirler. RAG, bu eksikliği gidermek için bilgi erişimi ve oluşturma aşamalarını birleştirerek daha doğru ve güvenilir çıktılar üretir.

## RAG Konfigürasyonları
RAG'ın çeşitli konfigürasyonları mevcuttur. Doğru RAG konfigürasyonunun seçilmesi, görevin niteliğine ve kullanılacak LLM modeline bağlıdır.

### Örnek Kod
Aşağıdaki örnek kod, RAG'ın basit bir uygulamasını göstermektedir. Bu örnekte, `transformers` kütüphanesinden `RagTokenizer` ve `RagSequenceForGeneration` kullanılmaktadır.

```python
from transformers import RagTokenizer, RagSequenceForGeneration

# Tokenizer ve modelin yüklenmesi
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

# Girdi metninin tokenleştirilmesi
input_dict = tokenizer.prepare_seq2seq_batch("What is RAG?", return_tensors="pt")

# Çıktının üretilmesi
generated_ids = model.generate(input_ids=input_dict["input_ids"], attention_mask=input_dict["attention_mask"])

# Üretilen çıktının çözülmesi
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_text)
```

### Kod Açıklaması
1. **Kütüphanelerin Yüklenmesi**: `transformers` kütüphanesinden `RagTokenizer` ve `RagSequenceForGeneration` sınıfları içe aktarılır.
2. **Tokenizer ve Modelin Yüklenmesi**: `facebook/rag-sequence-nq` modeli için önceden eğitilmiş `RagTokenizer` ve `RagSequenceForGeneration` örnekleri oluşturulur.
3. **Girdi Metninin Tokenleştirilmesi**: `tokenizer.prepare_seq2seq_batch` methodu kullanılarak girdi metni tokenleştirilir ve PyTorch tensorları olarak döndürülür.
4. **Çıktının Üretilmesi**: `model.generate` methodu kullanılarak, tokenleştirilmiş girdi metnine karşılık çıktı üretilir.
5. **Üretilen Çıktının Çözülmesi**: `tokenizer.batch_decode` methodu kullanılarak, üretilen çıktı IDs'leri metne çevrilir.

Bu kod, RAG'ın nasıl kullanılabileceğini gösteren basit bir örnektir. Gerçek dünya uygulamalarında, daha karmaşık konfigürasyonlar ve ince ayarlar gerekebilir.

---

## Naïve, advanced, and modular RAG configurations

## Naïve, Advanced, ve Modüler RAG (Retrieval-Augmented Generation) Konfigürasyonları

Bir RAG (Retrieval-Augmented Generation) çerçevesi (framework) mutlaka iki ana bileşen içerir: bir retriever (bulucu) ve bir generator (üretici). Generator, GPT-4o, Gemini, Llama veya başlangıç mimarilerinin yüzlerce varyasyonundan biri gibi herhangi bir LLM (Large Language Model) veya temel multimodal AI platformu veya modeli olabilir. Retriever, Activeloop, Pinecone, LlamaIndex, LangChain, Chroma ve daha fazlası gibi ortaya çıkan çerçeveler, yöntemler ve araçlardan herhangi biri olabilir. Şimdi sorun, hangi üç tip RAG çerçevesinin (Gao ve diğerleri, 2024) bir projenin ihtiyaçlarına uygun olacağına karar vermektir.

## Naïve RAG (Basit RAG)

Naïve RAG: Bu tür bir RAG çerçevesi, karmaşık veri embedding (gömme) ve indeksleme (indexing) içermez. Kullanıcı girdisini (input) artırmak (augment) ve tatmin edici bir yanıt elde etmek için anahtar kelimeler aracılığıyla makul miktarda veriye erişmek için verimli olabilir.

## Advanced RAG (Gelişmiş RAG)

Advanced RAG: Bu tür bir RAG, vektör arama (vector search) ve indeks-tabanlı retrieval (indexed-base retrieval) gibi daha karmaşık senaryolar içerir. Advanced RAG, çok çeşitli yöntemlerle uygulanabilir. Yapılandırılmış veya yapılandırılmamış, multimodal veriler dahil olmak üzere birden fazla veri türünü işleyebilir.

## Modüler RAG (Modular RAG)

Modüler RAG: Modüler RAG, naïve RAG, advanced RAG, makine öğrenimi (machine learning) ve karmaşık bir projeyi tamamlamak için gereken herhangi bir algoritmayı içeren her türlü senaryoyu kapsayacak şekilde ufku genişletir.

### Örnek Kod: Naïve, Advanced ve Modüler RAG

Aşağıdaki örnek kod, LangChain kütüphanesini kullanarak Naïve RAG, Advanced RAG ve Modüler RAG'ı göstermektedir.

```python
# Import gerekli kütüphaneler
from langchain import OpenAI, LLMChain
from langchain.retrievers import PineconeRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# Naïve RAG örneği
def naive_rag(query):
    # LLM modelini tanımla
    llm = OpenAI(model_name="text-davinci-003")
    # Prompt template tanımla
    prompt = PromptTemplate(input_variables=["query"], template="Query: {query}")
    # LLMChain oluştur
    chain = LLMChain(llm=llm, prompt=prompt)
    # Sorguyu çalıştır
    result = chain.run(query=query)
    return result

# Advanced RAG örneği
def advanced_rag(query):
    # Embedding modelini tanımla
    embeddings = OpenAIEmbeddings(model_name="text-embedding-ada-002")
    # Pinecone retriever tanımla
    retriever = PineconeRetriever(index_name="my_index", embeddings=embeddings)
    # İlgili belgeleri bul
    docs = retriever.get_relevant_documents(query)
    # LLM modelini tanımla
    llm = OpenAI(model_name="text-davinci-003")
    # Prompt template tanımla
    prompt = PromptTemplate(input_variables=["query", "docs"], template="Query: {query}\nDocs: {docs}")
    # LLMChain oluştur
    chain = LLMChain(llm=llm, prompt=prompt)
    # Sorguyu çalıştır
    result = chain.run(query=query, docs=docs)
    return result

# Modüler RAG örneği
def modular_rag(query):
    # Embedding modelini tanımla
    embeddings = OpenAIEmbeddings(model_name="text-embedding-ada-002")
    # Pinecone retriever tanımla
    retriever = PineconeRetriever(index_name="my_index", embeddings=embeddings)
    # İlgili belgeleri bul
    docs = retriever.get_relevant_documents(query)
    # Makine öğrenimi modelini tanımla (örneğin, bir sınıflandırıcı)
    # classifier = ...
    # LLM modelini tanımla
    llm = OpenAI(model_name="text-davinci-003")
    # Prompt template tanımla
    prompt = PromptTemplate(input_variables=["query", "docs"], template="Query: {query}\nDocs: {docs}")
    # LLMChain oluştur
    chain = LLMChain(llm=llm, prompt=prompt)
    # Sorguyu çalıştır
    result = chain.run(query=query, docs=docs)
    # Makine öğrenimi modelini kullanarak sonucu işle
    # result = classifier(result)
    return result

# Kullanım örneği
query = "örnek sorgu"
print(naive_rag(query))
print(advanced_rag(query))
print(modular_rag(query))
```

### Kod Açıklaması

Yukarıdaki kod, LangChain kütüphanesini kullanarak Naïve RAG, Advanced RAG ve Modüler RAG'ı göstermektedir.

*   `naive_rag` fonksiyonu, basit bir RAG örneğini gösterir. Kullanıcı girdisini alır ve bir LLM modeline gönderir.
*   `advanced_rag` fonksiyonu, gelişmiş bir RAG örneğini gösterir. Kullanıcı girdisini alır, ilgili belgeleri bulmak için bir Pinecone retriever kullanır ve daha sonra bir LLM modeline gönderir.
*   `modular_rag` fonksiyonu, modüler bir RAG örneğini gösterir. Kullanıcı girdisini alır, ilgili belgeleri bulmak için bir Pinecone retriever kullanır, makine öğrenimi modelini kullanarak sonucu işler ve daha sonra bir LLM modeline gönderir.

Her bir fonksiyon, farklı bir RAG yaklaşımını temsil eder ve kullanım örneğine göre seçilebilir.

### Önemli Noktalar

*   RAG çerçeveleri, retriever ve generator bileşenlerinden oluşur.
*   Naïve RAG, basit ve hızlıdır, ancak karmaşık senaryolar için uygun olmayabilir.
*   Advanced RAG, vektör arama ve indeks-tabanlı retrieval gibi daha karmaşık senaryolar içerir.
*   Modüler RAG, naïve RAG, advanced RAG, makine öğrenimi ve diğer algoritmaları içeren her türlü senaryoyu kapsayacak şekilde ufku genişletir.
*   RAG çerçeveleri, LangChain, Pinecone, LlamaIndex gibi kütüphaneler kullanılarak uygulanabilir.

---

## RAG versus fine-tuning

## RAG ve Fine-Tuning Karşılaştırması
RAG (Retrieval-Augmented Generation), her zaman fine-tuning'in alternatifi değildir ve fine-tuning de her zaman RAG'ın yerini alamaz. RAG veri kümelerinde çok fazla veri biriktiğinde, sistem yönetilmesi zor hale gelebilir. Öte yandan, günlük hava tahminleri, hisse senedi piyasa değerleri, şirket haberleri ve günlük olaylar gibi dinamik ve sürekli değişen verilerle bir modeli fine-tune edemeyiz. Bir modelin RAG veya fine-tune edilip edilmeyeceğine dair karar, parametrik (parametric) ve non-parametrik (non-parametric) bilgi oranına bağlıdır.

## Parametrik ve Non-Parametrik Bilgi
RAG tarafından desteklenen bir üretken yapay zeka ekosisteminde (generative AI ecosystem), parametrik kısım, eğitim verileri aracılığıyla öğrenilen model parametrelerine (ağırlıklara) atıfta bulunur. Bu, modelin bilgisinin bu öğrenilen ağırlıklarda ve biaslarda depolandığı anlamına gelir. Orijinal eğitim verileri matematiksel bir forma dönüştürülür ve buna parametrik temsil (parametric representation) denir. Esasen, model verilerden ne öğrendiğini "hatırlar", ancak verilerin kendisi açıkça depolanmaz.

## Non-Parametrik Bilgi
Buna karşılık, bir RAG ekosisteminin non-parametrik kısmı, doğrudan erişilebilen açık verileri depolamayı içerir. Bu, verilerin her zaman kullanılabilir olduğu ve gerektiğinde sorgulanabileceği anlamına gelir. Bilginin ağırlıklara dolaylı olarak gömülü olduğu parametrik modellerin aksine, RAG'daki non-parametrik veriler, her çıktı için gerçek verileri görmemize ve kullanmamıza olanak tanır.

## RAG ve Fine-Tuning Arasındaki Fark
RAG ve fine-tuning arasındaki fark, üretken yapay zeka modelinin işlemesi gereken statik (parametrik) ve dinamik (non-parametrik) sürekli değişen verilerin miktarına bağlıdır. Aşırı derecede RAG'a güvenen bir sistem yönetilmesi zor ve karmaşık hale gelebilir. Aşırı derecede fine-tuning'e güvenen bir sistem ise günlük bilgi güncellemelerine uyum sağlayamama sorunuyla karşılaşacaktır.

## Karar Verme Eşiği
Şekil 1.2'de gösterildiği gibi, bir RAG tarafından desteklenen üretken bir yapay zeka projesi yöneticisi, non-parametrik (açık veri) RAG çerçevesini uygulamadan önce ekosistemin eğitilmiş parametrik üretken yapay zeka modelinin potansiyelini değerlendirmelidir. RAG bileşeninin potansiyeli de dikkatlice değerlendirilmelidir.

## RAG ve Fine-Tuning Birlikte Kullanılabilir
Sonuç olarak, bir RAG tarafından desteklenen üretken bir yapay zeka ekosisteminde retriever ve generator arasındaki denge, projenin özel gereksinimlerine ve hedeflerine bağlıdır. RAG ve fine-tuning birbirini dışlamaz. RAG, bir modelin genel verimliliğini artırmak için kullanılabilir ve fine-tuning, RAG çerçevesi içindeki hem retrieval hem de generation bileşenlerinin performansını artırmak için bir yöntem olarak hizmet eder.

## Örnek Kod
Aşağıdaki örnek kod, RAG ve fine-tuning'in nasıl birlikte kullanılabileceğini gösterir:
```python
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Veri kümesini yükleme
train_data = pd.read_csv("train.csv")

# Model ve tokenizer'ı yükleme
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# RAG için retriever ve generator'u tanımlama
class Retriever(torch.nn.Module):
    def __init__(self):
        super(Retriever, self).__init__()
        self.encoder = model.encoder

    def forward(self, input_ids):
        outputs = self.encoder(input_ids)
        return outputs.last_hidden_state[:, 0, :]

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.decoder = model.decoder

    def forward(self, input_ids, encoder_outputs):
        outputs = self.decoder(input_ids, encoder_outputs=encoder_outputs)
        return outputs

# Fine-tuning için model'i tanımlama
class FineTuningModel(torch.nn.Module):
    def __init__(self):
        super(FineTuningModel, self).__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        return outputs

# RAG ve fine-tuning'i birlikte kullanma
def train():
    retriever = Retriever()
    generator = Generator()
    fine_tuning_model = FineTuningModel()

    # Retriever ve generator'u eğitme
    retriever_optimizer = torch.optim.Adam(retriever.parameters(), lr=1e-5)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-5)

    for epoch in range(5):
        for batch in train_data:
            input_ids = tokenizer(batch["input_text"], return_tensors="pt").input_ids
            retriever_outputs = retriever(input_ids)
            generator_outputs = generator(input_ids, retriever_outputs)

            # Kaybı hesaplama
            loss = torch.nn.CrossEntropyLoss()(generator_outputs, batch["target_text"])

            # Retriever ve generator'u güncelleme
            retriever_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            loss.backward()
            retriever_optimizer.step()
            generator_optimizer.step()

    # Fine-tuning model'ini eğitme
    fine_tuning_optimizer = torch.optim.Adam(fine_tuning_model.parameters(), lr=1e-5)

    for epoch in range(5):
        for batch in train_data:
            input_ids = tokenizer(batch["input_text"], return_tensors="pt").input_ids
            outputs = fine_tuning_model(input_ids)

            # Kaybı hesaplama
            loss = torch.nn.CrossEntropyLoss()(outputs, batch["target_text"])

            # Fine-tuning model'ini güncelleme
            fine_tuning_optimizer.zero_grad()
            loss.backward()
            fine_tuning_optimizer.step()

# Kodun açıklaması:
# Yukarıdaki kod, RAG ve fine-tuning'in nasıl birlikte kullanılabileceğini gösterir.
# Retriever ve generator, RAG çerçevesi içinde tanımlanır ve eğitilir.
# Fine-tuning model'i, RAG çerçevesi içinde tanımlanır ve eğitilir.
# Retriever, generator ve fine-tuning model'i, aynı veri kümesi üzerinde eğitilir.
# Kod, PyTorch kütüphanesini kullanır ve T5 modelini temel alır.
```

---

## The RAG ecosystem

## RAG Ekosistemi
RAG-tabanlı üretken yapay zeka (Generative AI), birçok konfigürasyonda uygulanabilen bir çerçevedir (framework). RAG'ın çerçevesi, Şekil 1.3'te gösterildiği gibi geniş bir ekosistem içinde çalışır. Ancak, karşılaşılan birçok geri alma ve oluşturma çerçevesi (retrieval and generation frameworks) ne olursa olsun, hepsi aşağıdaki dört alan ve bunlarla ilgili sorulara indirgenir:

## Veri (Data)
- Veriler nereden geliyor? 
- Güvenilir mi? 
- Yeterli mi? 
- Telif hakkı, gizlilik ve güvenlik sorunları var mı?

## Depolama (Storage)
- Veriler işlenmeden önce veya sonra nasıl depolanacak? 
- Ne kadar veri depolanacak?

## Geri Alma (Retrieval)
- Kullanıcının girdisini üretken model için yeterli hale getirmeden önce doğru veriler nasıl geri alınacak? 
- Bir proje için hangi tür RAG çerçevesi başarılı olacak?

## Oluşturma (Generation)
- Seçilen RAG çerçevesine hangi üretken yapay zeka modeli uyacak?

Veri, depolama ve oluşturma alanları, seçilen RAG çerçevesinin türüne büyük ölçüde bağlıdır. Bu seçimi yapmadan önce, uyguladığımız ekosistemdeki parametrik ve parametrik olmayan bilginin oranını değerlendirmemiz gerekir.

## RAG Çerçevesi Bileşenleri
Şekil 1.3, uygulanan RAG türlerine bakılmaksızın ana bileşenleri içeren RAG çerçevesini temsil eder:
- **The Retriever (D)**: Veri toplama, işleme, depolama ve geri almayı işler.
- **The Generator (G)**: Girdi artırma (input augmentation), prompt mühendisliği (prompt engineering) ve oluşturmayı işler.
- **The Evaluator (E)**: Matematiksel metrikler, insan değerlendirmesi ve geri bildirimi işler.
- **The Trainer (T)**: İlk önceden eğitilmiş modeli ve model ince ayarını (fine-tuning) işler.

Bu dört bileşenin her biri, genel RAG-tabanlı üretken yapay zeka boru hattını (pipeline) oluşturan ilgili ekosistemlere dayanır. Aşağıdaki bölümlerde D, G, E ve T alanlarına değineceğiz.

Örnek bir kod yapısı aşağıdaki gibi olabilir:
```python
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics import accuracy_score

# Veri yükleme ve işleme (Data loading and processing)
data = pd.read_csv("data.csv")

# Retriever (D) için veri işleme
def process_data(data):
    # Veri ön işleme adımları
    processed_data = data.apply(lambda x: x.strip())
    return processed_data

processed_data = process_data(data)

# Generator (G) için girdi artırma ve oluşturma
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

def generate_text(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Evaluator (E) için değerlendirme metrikleri
def evaluate(generated_text, reference_text):
    accuracy = accuracy_score(reference_text, generated_text)
    return accuracy

# Trainer (T) için model ince ayarı
def fine_tune_model(model, data):
    # Model ince ayarı adımları
    model.fit(data)
    return model

# Kullanım örneği
input_text = "Örnek girdi metni"
generated_text = generate_text(input_text)
print(generated_text)
```
## Kod Açıklaması
- `process_data` fonksiyonu, verileri ön işleme adımlarını içerir. Bu örnekte basitçe veri temizleme işlemi yapılmıştır.
- `generate_text` fonksiyonu, `T5` modelini kullanarak girdi metninden yeni metin oluşturur. Burada `input_ids` girdi metninin token IDs'lerini temsil eder.
- `evaluate` fonksiyonu, oluşturulan metni referans metne göre değerlendirir. Bu örnekte doğruluk skoru (accuracy score) kullanılmıştır.
- `fine_tune_model` fonksiyonu, modeli belirli bir veri kümesi üzerinde ince ayarını yapar.

Bu kod yapısı, RAG ekosisteminin temel bileşenlerini göstermek amacıyla basit bir örnek olarak tasarlanmıştır. Gerçek dünya uygulamalarında, daha karmaşık ve özelleştirilmiş çözümler gerekebilir.

---

## The retriever (D)

## Retriever (D) Bileşeni
Retriever (D) bileşeni, bir RAG (Retrieve, Augment, Generate) ekosisteminin veri toplama, işleme, depolama ve geri alma işlemlerini gerçekleştiren bileşenidir. RAG ekosisteminin başlangıç noktası, veri alımı (ingestion data) sürecidir ve ilk adım verilerin toplanmasıdır.

### Önemli Noktalar:
- Veri toplama (Data Collection)
- Veri işleme (Data Processing)
- Veri depolama (Data Storage)
- Veri geri alma (Data Retrieval)

## Veri Toplama ve İşleme Süreci
Veri toplama süreci, çeşitli kaynaklardan veri elde etmeyi içerir. Toplanan veriler daha sonra işlenir ve uygun bir formatta depolanır.

### Kod Örneği:
Aşağıdaki Python kodu, basit bir veri toplama ve işleme örneğini göstermektedir.
```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Veri toplama
data = {
    'text': ['Bu bir örnek cümledir.', 'Bu başka bir örnek cümledir.'],
    'label': [1, 0]
}

df = pd.DataFrame(data)

# Veri işleme
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(df['text'])

# Veri depolama
import pickle
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Veri geri alma
with open('tfidf.pkl', 'rb') as f:
    loaded_tfidf = pickle.load(f)
```
### Kod Açıklaması:
- `import pandas as pd`: Pandas kütüphanesini içe aktarır, veri çerçeveleri (DataFrame) oluşturmak için kullanılır.
- `import numpy as np`: NumPy kütüphanesini içe aktarır, sayısal işlemler için kullanılır.
- `from sklearn.feature_extraction.text import TfidfVectorizer`: Scikit-learn kütüphanesinden TfidfVectorizer sınıfını içe aktarır, metin verilerini TF-IDF (Term Frequency-Inverse Document Frequency) vektörlerine dönüştürmek için kullanılır.
- `data = {...}`: Örnek veri kümesini tanımlar.
- `df = pd.DataFrame(data)`: Veri kümesini bir Pandas DataFrame'e dönüştürür.
- `vectorizer = TfidfVectorizer()`: TfidfVectorizer nesnesi oluşturur.
- `tfidf = vectorizer.fit_transform(df['text'])`: DataFrame'deki 'text' sütununu TF-IDF vektörlerine dönüştürür.
- `import pickle`: Pickle kütüphanesini içe aktarır, Python nesnelerini serileştirmek ve deserialize etmek için kullanılır.
- `with open('tfidf.pkl', 'wb') as f:`: 'tfidf.pkl' dosyasını yazma modunda açar ve TF-IDF vektörlerini bu dosyaya yazar.
- `with open('tfidf.pkl', 'rb') as f:`: 'tfidf.pkl' dosyasını okuma modunda açar ve TF-IDF vektörlerini bu dosyadan okur.

## Sonuç
Retriever bileşeni, RAG ekosisteminin temel bileşenlerinden biridir ve veri toplama, işleme, depolama ve geri alma işlemlerini gerçekleştirir. Yukarıdaki kod örneği, basit bir veri toplama ve işleme sürecini göstermektedir.

---

## Collect (D1)

## Veri Toplanması (Data Collection - D1)
Günümüz dünyasında, yapay zeka verileri (AI data) medya çalma listelerimiz kadar çeşitlidir. Bu veriler, bir blog gönderisindeki metin parçasından bir memeye (meme) veya kulaklıklar aracılığıyla akış yapılan en son hit şarkıya kadar her şey olabilir. Ve bu verilerin dosya formatları da çeşitli şekil ve boyutlardadır. Detaylarla dolu PDF'ler, web sayfaları, düz metin dosyaları, düzenli JSON dosyaları, MP3 müzikleri, MP4 formatındaki videolar veya PNG ve JPG formatındaki resimlere kadar birçok farklı veri türü mevcuttur. Ayrıca, bu verilerin büyük bir kısmı yapılandırılmamış (unstructured) ve karmaşık şekillerde bulunur. Neyse ki, Pinecone, OpenAI, Chroma ve Activeloop gibi birçok platform, bu veri ormanını işlemek ve depolamak için kullanıma hazır araçlar sağlar.

### Önemli Noktalar:
- Veri çeşitliliği (Data Diversity)
- Yapılandırılmamış veri (Unstructured Data)
- Farklı dosya formatları (Various File Formats)
- Veri işleme ve depolama platformları (Data Processing and Storage Platforms)

## Kullanılan Kodlar ve Açıklamaları
Veri toplama ve işleme sürecinde kullanılan bazı kod örnekleri aşağıda verilmiştir. Bu örneklerde Python programlama dili kullanılmıştır.

### JSON Veri Okuma
```python
import json

# JSON dosyasını açma ve okuma
with open('data.json', 'r') as file:
    data = json.load(file)

print(data)
```
Bu kod, `data.json` adlı bir JSON dosyasını açar ve içeriğini `data` değişkenine yükler. JSON dosyaları, veri depolama ve değişim için yaygın olarak kullanılan bir formattır.

### Metin Dosyasını Okuma
```python
# Metin dosyasını açma ve okuma
with open('example.txt', 'r') as file:
    text = file.read()

print(text)
```
Bu kod, `example.txt` adlı bir metin dosyasını okur ve içeriğini `text` değişkenine atar.

### Resim Dosyasını Okuma (Pillow Kütüphanesi Kullanılarak)
```python
from PIL import Image

# Resmi açma
img = Image.open('image.jpg')

# Resmin boyutlarını yazdırma
print(img.size)
```
Bu kod, `image.jpg` adlı bir resim dosyasını açar ve boyutlarını yazdırır. Pillow kütüphanesi, resim işleme için kullanılan popüler bir Python kütüphanesidir.

### MP3 Dosyasını Okuma (Mutagen Kütüphanesi Kullanılarak)
```python
from mutagen.mp3 import MP3

# MP3 dosyasını açma
audio = MP3('song.mp3')

# MP3 dosyasının süresini yazdırma
print(audio.info.length)
```
Bu kod, `song.mp3` adlı bir MP3 dosyasını açar ve süresini yazdırır. Mutagen kütüphanesi, ses dosyaları hakkında meta veri okumak için kullanılır.

### PDF Dosyasını Okuma (PyPDF2 Kütüphanesi Kullanılarak)
```python
import PyPDF2

# PDF dosyasını açma
with open('document.pdf', 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()

print(text)
```
Bu kod, `document.pdf` adlı bir PDF dosyasını açar ve içeriğini metin olarak çıkarır. PyPDF2 kütüphanesi, PDF dosyalarını okumak ve işlemek için kullanılır.

Tüm bu kod örnekleri, farklı veri türlerini işlemek için kullanılan çeşitli Python kütüphanelerini ve tekniklerini göstermektedir. Veri toplama ve işleme sürecinde, bu tür kütüphaneler ve teknikler büyük önem taşır.

---

## Process (D2)

## Çok Modlu Veri İşleme (Multimodal Data Processing) Süreci (D2)

Çok modlu veri işleme sürecinin veri toplama aşamasında (D1), metin, resim ve video gibi çeşitli veri türleri, web kazıma teknikleri (web scraping techniques) veya diğer bilgi kaynaklarından elde edilebilir. Bu veri nesneleri daha sonra tek tip özellik temsilleri (uniform feature representations) oluşturmak için dönüştürülür. Örneğin, veriler daha küçük parçalara ayrılabilir (chunked), vektörlere dönüştürülebilir (embedded) ve aranabilirlik ile erişim verimliliğini artırmak için indekslenebilir (indexed).

## Veri Dönüştürme Teknikleri

Bu teknikler, verilerin daha etkili bir şekilde işlenmesini ve kullanılmasını sağlar. 
- **Parçalama (Chunking)**: Büyük veri parçalarını daha küçük ve yönetilebilir parçalara ayırma işlemidir.
- **Gömme (Embedding)**: Verileri vektör uzayında temsil etme işlemidir. Bu, metin, resim veya diğer veri türlerinin sayısal vektörlere dönüştürülmesini sağlar.
- **İndeksleme (Indexing)**: Verileri daha hızlı erişim ve arama için organize etme işlemidir.

## Python'da Hibrit Adaptif RAG Oluşturma

Bu bölümde, Python kullanarak Hibrit Adaptif RAG (Hybrid Adaptive RAG) oluşturmaya başlayacağız. RAG, Retrieval-Augmented Generation (Alınan-Augmentasyonlu Üretim) anlamına gelir ve bilgi erişimini ve üretken modelleri birleştirir.

### Örnek Kod

```python
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch

# Veri yükleme ve ön işleme
data = pd.read_csv("data.csv")

# Model ve tokenizer yükleme
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Metin gömme (text embedding)
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.detach().numpy()[0]

# Örnek metin gömme
text = "Bu bir örnek metindir."
embedding = embed_text(text)
print(embedding)

# İndeksleme için örnek kod
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Vektörler arasında benzerlik ölçümü
vector1 = np.array([1, 2, 3]).reshape(1, -1)
vector2 = np.array([4, 5, 6]).reshape(1, -1)
similarity = cosine_similarity(vector1, vector2)
print("Benzerlik:", similarity)
```

### Kod Açıklaması

1. **Veri Yükleme**: `pd.read_csv("data.csv")` ile veri seti yüklenir.
2. **Model ve Tokenizer Yükleme**: `AutoModel` ve `AutoTokenizer` kullanılarak önceden eğitilmiş model ve tokenizer yüklenir.
3. **Metin Gömme**: `embed_text` fonksiyonu, metni vektör temsiline dönüştürür. Bu işlem, `tokenizer` ile metni tokenlara ayırma, `model` ile tokenları işleme ve son gizli katmandan vektör elde etmeyi içerir.
4. **İndeksleme**: Örnek kod, vektörler arasında benzerlik ölçümü için `cosine_similarity` kullanır. Bu, indeksleme ve benzerlik tabanlı arama için temel bir adımdır.

## İleri Veri İşleme Fonksiyonları

İleriki bölümlerde, daha karmaşık veri işleme fonksiyonları oluşturmaya devam edeceğiz. Bu, çok modlu veri işleme sürecinin daha ileri aşamalarını içerecektir.

---

## Storage (D3)

## Depolama (D3)
Bu işlem hattının bu aşamasında, internetten toplanmış ve işlenmeye başlanmış büyük miktarda çeşitli veriye sahibiz - videolar, resimler, metinler ve daha fazlası. Şimdi, bu verileri yararlı hale getirmek için ne yapabiliriz? İşte burada Deep Lake, Pinecone ve Chroma gibi vektör depoları devreye giriyor. Bunları, verilerinizi sadece depolamakla kalmayıp aynı zamanda matematiksel varlıklar olarak vektörlere dönüştüren süper akıllı kütüphaneler olarak düşünün, böylece güçlü hesaplamalar mümkün olur. Ayrıca hızlı erişim için çeşitli indeksleme yöntemleri ve diğer teknikleri de uygulayabilirler.

## Vektör Depolarının Önemi
Vektör depoları, verileri statik elektronik tablolarda ve dosyalarda tutmak yerine, sohbet botlarından arama motorlarına kadar her şeyi güçlendirebilecek dinamik, aranabilir bir sisteme dönüştürür.

## Vektör Depolarının Kullanımı
Deep Lake, Pinecone ve Chroma gibi vektör depoları, verileri vektörlere dönüştürerek güçlü hesaplamalar yapabilmemizi sağlar. Bu sayede, verilerimizi daha etkili bir şekilde işleyebilir ve kullanabiliriz.

## Örnek Kod
Aşağıdaki örnek kod, Deep Lake vektör deposunu kullanarak veri depolama ve sorgulama işlemlerini göstermektedir:
```python
import numpy as np
from deeplake import Dataset

# Verileri oluştur
data = np.random.rand(100, 128)

# Deep Lake veri kümesi oluştur
ds = Dataset("my_dataset")

# Verileri vektör olarak depola
ds.create_tensor("vectors", htype="generic", dtype=np.float32)
ds.vectors.extend(data)

# Verileri sorgula
query_vector = np.random.rand(1, 128)
results = ds.vectors.search(query_vector, k=5)

# Sonuçları yazdır
print(results)
```
## Kod Açıklaması
Yukarıdaki kod, Deep Lake kütüphanesini kullanarak bir veri kümesi oluşturur ve verileri vektör olarak depolar. Daha sonra, bir sorgu vektörü oluşturur ve bu vektöre en yakın 5 vektörü bulur.

*   `import numpy as np`: Numpy kütüphanesini içe aktarır.
*   `from deeplake import Dataset`: Deep Lake kütüphanesinden `Dataset` sınıfını içe aktarır.
*   `data = np.random.rand(100, 128)`: 100 adet 128 boyutlu rastgele vektör oluşturur.
*   `ds = Dataset("my_dataset")`: "my_dataset" adında bir Deep Lake veri kümesi oluşturur.
*   `ds.create_tensor("vectors", htype="generic", dtype=np.float32)`: "vectors" adında bir tensor oluşturur ve veri tipini `np.float32` olarak belirler.
*   `ds.vectors.extend(data)`: Oluşturulan verileri "vectors" tensoruna ekler.
*   `query_vector = np.random.rand(1, 128)`: Bir sorgu vektörü oluşturur.
*   `results = ds.vectors.search(query_vector, k=5)`: Sorgu vektörüne en yakın 5 vektörü bulur.
*   `print(results)`: Sonuçları yazdırır.

## Avantajları
Vektör depoları, verileri daha etkili bir şekilde depolamamızı ve sorgulamamızı sağlar. Bu sayede, daha hızlı ve doğru sonuçlar elde edebiliriz.

## Kullanım Alanları
Vektör depoları, sohbet botlarından arama motorlarına kadar birçok alanda kullanılabilir.

*   Sohbet Botları (Chatbots)
*   Arama Motorları (Search Engines)
*   Öneri Sistemleri (Recommendation Systems)
*   Görüntü ve Video İşleme (Image and Video Processing)

## Sonuç
Vektör depoları, verileri daha etkili bir şekilde depolamamızı ve sorgulamamızı sağlayan güçlü bir araçtır. Deep Lake, Pinecone ve Chroma gibi vektör depoları, verileri vektörlere dönüştürerek güçlü hesaplamalar yapabilmemizi sağlar. Bu sayede, daha hızlı ve doğru sonuçlar elde edebiliriz.

---

## Retrieval query (D4)

## Veri Erişim Sorgusu (D4)
Veri erişim süreci, kullanıcı girişi veya otomatik giriş (G1) tarafından tetiklenir. Verileri hızlı bir şekilde erişmek için, uygun bir formata dönüştürdükten sonra vektör depolarına ve veri kümelerine yüklüyoruz. Ardından, anahtar kelime aramaları, akıllı embeddings (embeddings) ve indeksleme (indexing) kombinasyonunu kullanarak verileri etkin bir şekilde erişiyoruz. Örneğin, Kosinüs benzerliği (Cosine similarity), birbirine yakın ilişkili öğeleri bulur ve arama sonuçlarının sadece hızlı değil, aynı zamanda yüksek ilgili olmasını sağlar. Veriler erişildikten sonra, girişi zenginleştiriyoruz (augment the input).

## Önemli Noktalar:
* Veri erişim süreci, kullanıcı girişi veya otomatik giriş tarafından tetiklenir (G1).
* Veriler, uygun bir formata dönüştürüldükten sonra vektör depolarına ve veri kümelerine yüklenir.
* Anahtar kelime aramaları, akıllı embeddings ve indeksleme kombinasyonu kullanılır.
* Kosinüs benzerliği, birbirine yakın ilişkili öğeleri bulur.
* Veriler erişildikten sonra, girişi zenginleştirme işlemi yapılır.

## Kullanılan Kodlar ve Açıklamalar:
Veri erişim sorgusu için kullanılan kodlar genellikle Python programlama dilinde yazılır ve çeşitli kütüphaneler kullanılır. Aşağıdaki örnek kod, veri erişim sorgusu için kullanılan bazı temel kavramları içerir:
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

# Veri kümesini yükleme
dataset = load_dataset("path/to/dataset")

# Vektör deposunu oluşturma
vector_store = {}

# Veri kümesindeki her bir örnek için vektör oluşturma
for example in dataset:
    inputs = tokenizer(example["text"], return_tensors="pt")
    outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].detach().numpy()
    vector_store[example["id"]] = vector

# Kosinüs benzerliği kullanarak benzer öğeleri bulma
def find_similar_items(query_vector, vector_store, top_n=5):
    similarities = []
    for id, vector in vector_store.items():
        similarity = cosine_similarity(query_vector, vector)
        similarities.append((id, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Sorgu vektörünü oluşturma
query = "örnek sorgu metni"
query_inputs = tokenizer(query, return_tensors="pt")
query_outputs = model(**query_inputs)
query_vector = query_outputs.last_hidden_state[:, 0, :].detach().numpy()

# Benzer öğeleri bulma
similar_items = find_similar_items(query_vector, vector_store)

# Sonuçları yazdırma
for id, similarity in similar_items:
    print(f"ID: {id}, Benzerlik: {similarity:.4f}")
```
## Kod Açıklamaları:
* `import numpy as np`: Numpy kütüphanesini içe aktarır.
* `from sklearn.metrics.pairwise import cosine_similarity`: Scikit-learn kütüphanesinden Kosinüs benzerliği fonksiyonunu içe aktarır.
* `from transformers import AutoModel, AutoTokenizer`: Transformers kütüphanesinden otomatik model ve tokenizer'ı içe aktarır.
* `dataset = load_dataset("path/to/dataset")`: Veri kümesini yükler.
* `vector_store = {}`: Vektör deposunu oluşturur.
* `for example in dataset:`: Veri kümesindeki her bir örnek için vektör oluşturur.
* `inputs = tokenizer(example["text"], return_tensors="pt")`: Örnek metni tokenize eder ve PyTorch tensörleri olarak döndürür.
* `outputs = model(**inputs)`: Tokenize edilmiş metni modele geçirir ve çıktıları alır.
* `vector = outputs.last_hidden_state[:, 0, :].detach().numpy()`: Çıktıların son gizli katmanını alır ve numpy dizisi olarak döndürür.
* `vector_store[example["id"]] = vector`: Vektörü vektör deposuna kaydeder.
* `def find_similar_items(query_vector, vector_store, top_n=5):`: Kosinüs benzerliği kullanarak benzer öğeleri bulur.
* `similarities = []`: Benzerlikleri saklamak için bir liste oluşturur.
* `for id, vector in vector_store.items():`: Vektör deposundaki her bir vektör için Kosinüs benzerliğini hesaplar.
* `similarity = cosine_similarity(query_vector, vector)`: Kosinüs benzerliğini hesaplar.
* `similarities.append((id, similarity))`: Benzerliği listeye ekler.
* `similarities.sort(key=lambda x: x[1], reverse=True)`: Benzerlikleri sıralar.
* `return similarities[:top_n]`: En benzer `top_n` öğeyi döndürür.
* `query = "örnek sorgu metni"`: Sorgu metni oluşturur.
* `query_inputs = tokenizer(query, return_tensors="pt")`: Sorgu metni tokenize eder.
* `query_outputs = model(**query_inputs)`: Tokenize edilmiş sorgu metni modele geçirir.
* `query_vector = query_outputs.last_hidden_state[:, 0, :].detach().numpy()`: Sorgu vektörünü oluşturur.
* `similar_items = find_similar_items(query_vector, vector_store)`: Benzer öğeleri bulur.
* `for id, similarity in similar_items:`: Sonuçları yazdırır.

## İngilizce Teknik Terimler:
* Veri erişim sorgusu: Retrieval query (D4)
* Vektör deposu: Vector store
* Anahtar kelime aramaları: Keyword searches
* Akıllı embeddings: Smart embeddings
* İndeksleme: Indexing
* Kosinüs benzerliği: Cosine similarity
* Girişi zenginleştirme: Augment the input

---

## The generator (G)

## Jeneratör (G)

RAG ekosisteminde girdi (input) ve geri alma (retrieval) arasındaki sınırlar bulanıktır, Şekil 1.3'te gösterildiği gibi. Şekil 1.3, RAG çerçevesini ve ekosistemini temsil etmektedir. Kullanıcı girişi (G1), otomatik veya insan tarafından, üretken modele göndermeden önce girişi zenginleştirmek için geri alma sorgusu (D4) ile etkileşime girer. Üretken akış bir girdi ile başlar.

## Önemli Noktalar
* Kullanıcı girişi (G1) ve geri alma sorgusu (D4) arasındaki etkileşim RAG ekosisteminin temelini oluşturur.
* Giriş, üretken modele gönderilmeden önce zenginleştirilir (augment).
* Üretken akış (generative flow) bir girdi ile başlar.

## Kod Örneği
RAG ekosistemini temsil eden kod örneği aşağıdaki gibidir:
```python
import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Kullanıcı girişi (G1)
input_text = "örnek girdi"

# Geri alma sorgusu (D4)
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq")

# Tokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

# Girişleri tokenleştirme
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Geri alma sorgusunu çalıştırma
retrieved_docs = retriever(input_ids, return_tensors="pt")

# Üretken model
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

# Üretken akışı başlatma
output = model.generate(input_ids, retrieved_docs)

# Çıktıyı çözme
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output)
```
## Kod Açıklaması

* `import torch`: PyTorch kütüphanesini içe aktarır.
* `from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration`: Transformers kütüphanesinden RAG tokenizer, retriever ve sequence for generation sınıflarını içe aktarır.
* `input_text = "örnek girdi"`: Kullanıcı girişini tanımlar.
* `retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq")`: Geri alma sorgusunu önceden eğitilmiş model ile başlatır.
* `tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")`: Tokenizer'ı önceden eğitilmiş model ile başlatır.
* `input_ids = tokenizer(input_text, return_tensors="pt").input_ids`: Girişleri tokenleştirir ve PyTorch tensörleri olarak döndürür.
* `retrieved_docs = retriever(input_ids, return_tensors="pt")`: Geri alma sorgusunu çalıştırır ve PyTorch tensörleri olarak döndürür.
* `model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")`: Üretken modeli önceden eğitilmiş model ile başlatır.
* `output = model.generate(input_ids, retrieved_docs)`: Üretken akışı başlatır ve çıktı üretir.
* `decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)`: Çıktıyı çözer ve özel tokenleri atlar.

## Kullanım
Bu kod örneği, RAG ekosistemini kullanarak bir girdi ve geri alma sorgusu ile üretken akışı başlatır. Kullanıcı girişi ve geri alma sorgusu arasındaki etkileşimi gösterir ve üretken modelin çıktısını üretir.

---

## Input (G1)

## Giriş (G1)
Giriş, otomatik görevlerin bir toplu işlenmesi (örneğin e-postaların işlenmesi) veya Kullanıcı Arayüzü (UI) aracılığıyla insan tarafından verilen komutlar olabilir. Bu esneklik, çeşitli profesyonel ortamlara Yapay Zeka (AI) 'nı sorunsuz bir şekilde entegre etmenize olanak tanır ve endüstriler genelinde üretkenliği artırır.

## Önemli Noktalar
* Giriş otomatik görevler veya insan komutları olabilir (Human Prompts).
* Kullanıcı Arayüzü (UI) insan komutları için kullanılır.
* Yapay Zeka (AI) çeşitli profesyonel ortamlara entegre edilebilir.
* Üretkenlik endüstriler genelinde artırılabilir.

## Kod Örneği
Aşağıdaki kod örneği, bir Kullanıcı Arayüzü (UI) oluşturmak için kullanılan Python kodudur:
```python
import tkinter as tk
from tkinter import messagebox

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Merhaba"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")

        self.quit = tk.Button(self, text="Çıkış", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def say_hi(self):
        print("Merhaba!")

root = tk.Tk()
app = Application(master=root)
app.mainloop()
```
## Kod Açıklaması
* `import tkinter as tk`: Tkinter kütüphanesini içe aktarır ve `tk` takma adını verir. (Tkinter is a Python binding to the Tk GUI toolkit.)
* `from tkinter import messagebox`: Tkinter kütüphanesinden `messagebox` modülünü içe aktarır. (Messagebox is a module in Tkinter that provides functions to display message boxes.)
* `class Application(tk.Frame)`: `Application` adlı bir sınıf tanımlar ve `tk.Frame` sınıfından miras alır. (This defines a class `Application` that inherits from `tk.Frame`.)
* `def create_widgets(self):`: `create_widgets` adlı bir metot tanımlar ve içinde Kullanıcı Arayüzü (UI) öğelerini oluşturur. (This method creates the UI widgets.)
* `self.hi_there = tk.Button(self)`: Bir düğme (Button) oluşturur ve `self.hi_there` değişkenine atar. (This creates a button widget.)
* `self.hi_there["text"] = "Merhaba"`: Düğmenin metnini "Merhaba" olarak ayarlar. (This sets the text of the button to "Merhaba".)
* `self.hi_there["command"] = self.say_hi`: Düğmeye tıklandığında `say_hi` metodunu çağırmak üzere ayarlar. (This sets the command to be executed when the button is clicked.)
* `self.quit = tk.Button(self, text="Çıkış", fg="red", command=self.master.destroy)`: Bir çıkış düğmesi oluşturur ve `self.quit` değişkenine atar. (This creates a quit button.)
* `app.mainloop()`: Uygulamanın ana döngüsünü başlatır. (This starts the main event loop of the application.)

## Kullanım
Yukarıdaki kodu bir Python dosyasına kaydedin ve çalıştırın. Bir Kullanıcı Arayüzü (UI) penceresi açılacaktır. "Merhaba" düğmesine tıkladığınızda, konsola "Merhaba!" yazdırılacaktır. "Çıkış" düğmesine tıkladığınızda, uygulama kapanacaktır.

---

## Augmented input with HF (G2)

## Artırılmış Giriş ile İnsan Geri Bildirimi (HF - G2)
İnsan geri bildirimi (HF), Değerlendirici (E) altında İnsan Geri Bildirimi (E2) bölümünde açıklandığı gibi girdiye eklenebilir. İnsan geri bildirimi, bir RAG ekosistemini oldukça uyarlanabilir hale getirecek ve veri alma ve üretken yapay zeka girdileri üzerinde tam kontrol sağlayacaktır.

## İnsan Geri Bildirimi ile Artırılmış Giriş Nasıl Kullanılır?
İnsan geri bildirimi, bir RAG sisteminin daha esnek ve kontrol edilebilir olmasını sağlar. Bu sayede, sistemin performansı ve çıktı kalitesi artırılabilir.

## Önemli Noktalar
* İnsan geri bildirimi, RAG ekosistemini uyarlanabilir hale getirir (makes a RAG ecosystem considerably adaptable).
* Veri alma ve üretken yapay zeka girdileri üzerinde tam kontrol sağlar (provide full control over data retrieval and generative AI inputs).
* Artırılmış giriş ile insan geri bildirimi, Python'da hibrit uyarlanabilir RAG oluşturmada kullanılır (Building hybrid adaptive RAG in Python).

## Kod Örneği
Bu bölümde, artırılmış giriş ile insan geri bildirimi kullanılarak Python'da hibrit uyarlanabilir RAG oluşturma örneği verilmiştir. Ancak, metinde bir kod örneği bulunmamaktadır. 

Eğer bir kod örneği verecek olsaydık, aşağıdaki gibi bir yapı kullanabilirdik:
```python
# Import gerekli kütüphaneler
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Model ve tokenizer yükleme
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# İnsan geri bildirimi ile artırılmış giriş oluşturma
def create_augmented_input(input_text, human_feedback):
    # İnsan geri bildirimi ile giriş metnini birleştirme
    augmented_input = f"{input_text} {human_feedback}"
    return augmented_input

# Örnek kullanım
input_text = "Bu bir örnek metindir."
human_feedback = "Bu metin olumlu bir geri bildirim almıştır."
augmented_input = create_augmented_input(input_text, human_feedback)
print(augmented_input)
```
## Kod Açıklaması
Yukarıdaki kod örneğinde, `create_augmented_input` fonksiyonu, giriş metni ve insan geri bildirimini birleştirerek artırılmış giriş oluşturur. Bu fonksiyon, `input_text` ve `human_feedback` parametrelerini alır ve birleştirilmiş metni döndürür.

Bu kod, `transformers` kütüphanesini kullanarak bir BERT model ve tokenizer yükler. Ancak, bu kodun asıl amacı, artırılmış giriş ile insan geri bildirimini birleştirmektir.

## Not
Bu metinde, bir kod örneği bulunmamaktadır. Yukarıdaki kod örneği, sadece bir gösterimdir. Gerçek kod örneği, bağlam ve kullanılan kütüphanelere göre değişebilir.

---

## Prompt engineering (G3)

## Prompt Mühendisliği (Prompt Engineering)

Prompt mühendisliği, hem retriever (D) hem de generator (G) için standart ve artırılmış mesajları hazırlamak üzere kullanılan bir tekniktir (Technique). Bu teknik, üretken yapay zeka modelinin (Generative AI Model) işleyeceği verileri hazırlamak için kullanılır.

## Prompt Mühendisliğinin Önemi

*   Retriever'ın (D) çıktısını ve kullanıcı girdisini (User Input) bir araya getirir.
*   Üretken yapay zeka modelinin (Generative AI Model) daha doğru ve alakalı sonuçlar üretmesini sağlar.

## Prompt Mühendisliği ile İlgili Kod Örneği

Aşağıdaki kod örneğinde, prompt mühendisliğinin nasıl uygulanabileceği gösterilmektedir:
```python
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

## Model ve Tokenizer'ın Yüklenmesi
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

## Kullanıcı Girdisi (User Input)
user_input = "İstediğiniz bir metni buraya girin"

## Retriever Çıktısı (Retriever Output)
retriever_output = "Retriever'ın ürettiği metni buraya girin"

## Prompt Mühendisliği ile Metinlerin Birleştirilmesi
input_text = f"{user_input} {retriever_output}"

## Tokenization
input_ids = tokenizer.encode(input_text, return_tensors='pt')

## Model Çıktısı (Model Output)
output = model.generate(input_ids)

## Çıktının Metin Haline Getirilmesi
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
```
## Kod Açıklaması

*   `import torch`: PyTorch kütüphanesini içe aktarır (Import).
*   `from transformers import T5Tokenizer, T5ForConditionalGeneration`: Transformers kütüphanesinden T5Tokenizer ve T5ForConditionalGeneration sınıflarını içe aktarır.
*   `tokenizer = T5Tokenizer.from_pretrained('t5-small')`: T5Tokenizer'ı 't5-small' modelini kullanarak yükler (Load).
*   `model = T5ForConditionalGeneration.from_pretrained('t5-small')`: T5ForConditionalGeneration modelini 't5-small' modelini kullanarak yükler.
*   `user_input = "İstediğiniz bir metni buraya girin"`: Kullanıcı girdisini (User Input) tanımlar.
*   `retriever_output = "Retriever'ın ürettiği metni buraya girin"`: Retriever çıktısını (Retriever Output) tanımlar.
*   `input_text = f"{user_input} {retriever_output}"`: Prompt mühendisliği ile kullanıcı girdisini ve retriever çıktısını birleştirir (Concatenate).
*   `input_ids = tokenizer.encode(input_text, return_tensors='pt')`: Birleştirilen metni tokenleştirir (Tokenize) ve PyTorch tensor'u (Tensor) olarak döndürür.
*   `output = model.generate(input_ids)`: Model çıktısını (Model Output) üretir.
*   `generated_text = tokenizer.decode(output[0], skip_special_tokens=True)`: Model çıktısını metin haline getirir (Decode) ve özel tokenları (Special Tokens) atlar.

## Prompt Mühendisliğinin Faydaları

*   Üretken yapay zeka modellerinin (Generative AI Models) daha doğru ve alakalı sonuçlar üretmesini sağlar.
*   Kullanıcı girdisi (User Input) ve retriever çıktısını (Retriever Output) bir araya getirerek daha kapsamlı bir metin oluşturur.

## Sonuç

Prompt mühendisliği, üretken yapay zeka modellerinin (Generative AI Models) daha doğru ve alakalı sonuçlar üretmesini sağlayan önemli bir tekniktir (Technique). Kullanıcı girdisi (User Input) ve retriever çıktısını (Retriever Output) bir araya getirerek daha kapsamlı bir metin oluşturur.

---

## Generation and output (G4)

## G4: Üretken Yapay Zeka Modellerinin Seçimi ve Çıktı (Generation and Output)
Bir üretken yapay zeka (Generative AI) modelinin seçimi, projenin hedeflerine (project goals) bağlıdır. Llama, Gemini, GPT ve diğer modeller çeşitli gereksinimlere (requirements) uygun olabilir. Ancak, her modelin spesifikasyonlarına (model specifications) uygun bir şekilde prompt (prompt) tasarlanmalıdır. LangChain gibi çerçeveler (frameworks), çeşitli yapay zeka modellerini uygulamalarla entegre etmeyi kolaylaştırarak esnek arayüzler (adaptable interfaces) ve araçlar (tools) sağlar.

## Önemli Noktalar
* Üretken yapay zeka model seçimi projenin hedeflerine bağlıdır (project goals).
* Farklı modeller (Llama, Gemini, GPT) çeşitli gereksinimlere uygun olabilir.
* Her modelin spesifikasyonlarına uygun prompt tasarlanmalıdır.
* LangChain gibi çerçeveler entegrasyonu kolaylaştırır.

## Kullanılan Kodlar ve Açıklamalar
Bu bölümde LangChain çerçevesini kullanarak üretken yapay zeka modellerini entegre etmek için örnek bir kod verilmiştir.

```python
# Gerekli kütüphanelerin import edilmesi
from langchain import LLMChain
from langchain.llms import OpenAI

# OpenAI modelinin seçilmesi ve LLMChain nesnesinin oluşturulması
llm = OpenAI(model_name="text-davinci-003")
chain = LLMChain(llm=llm)

# Prompt tasarımı ve modele gönderilmesi
prompt = "Bir üretken yapay zeka modelinin seçimi nasıl yapılır?"
response = chain.run(prompt)

# Modelin çıktısının yazdırılması
print(response)
```

## Kod Açıklaması
* `from langchain import LLMChain`: LangChain kütüphanesinden `LLMChain` sınıfını import eder. Bu sınıf, bir dil modelini (LLM) kullanarak bir zincir (chain) oluşturmayı sağlar.
* `from langchain.llms import OpenAI`: LangChain kütüphanesinden `OpenAI` sınıfını import eder. Bu sınıf, OpenAI dil modelini kullanarak işlem yapmayı sağlar.
* `llm = OpenAI(model_name="text-davinci-003")`: OpenAI dil modelini seçer ve `llm` nesnesini oluşturur. `model_name` parametresi ile kullanılacak model belirlenir.
* `chain = LLMChain(llm=llm)`: `LLMChain` nesnesini oluşturur ve `llm` nesnesini parametre olarak verir.
* `prompt = "Bir üretken yapay zeka modelinin seçimi nasıl yapılır?"`: Prompt tasarlanır.
* `response = chain.run(prompt)`: Prompt, modele gönderilir ve yanıt alınır.
* `print(response)`: Modelin çıktısı yazdırılır.

## Kullanılan Teknik Terimler
* Üretken Yapay Zeka (Generative AI)
* LangChain (LangChain framework)
* OpenAI (OpenAI model)
* LLM (Large Language Model)
* Prompt (Prompt design)
* Zincir (Chain)

---

## The evaluator (E)

## Değerlendirici (E)
Bir üretken yapay zeka (Generative AI) modelinin performansını değerlendirmek için genellikle matematiksel metrikler (mathematical metrics) kullanırız. Ancak, bu metrikler bize sadece kısmi bir resim sunar. Bir yapay zeka'nın etkinliğinin nihai testinin insan değerlendirmesine (human evaluation) bağlı olduğunu hatırlamak önemlidir.

## Matematiksel Metriklerin Sınırlılıkları
Matematiksel metrikler, bir üretken yapay zeka modelinin performansını değerlendirmek için kullanılır, ancak bunlar tek başına yeterli değildir. Bu metrikler, modelin ürettiği çıktıların kalitesini (output quality) ve çeşitliliğini (diversity) ölçmede yetersiz kalabilir.

## İnsan Değerlendirmesinin Önemi
İnsan değerlendirmesi, bir üretken yapay zeka modelinin gerçek dünyadaki performansını değerlendirmek için gereklidir. İnsan değerlendiriciler, modelin ürettiği çıktıların kalitesini, anlamını ve bağlamını daha iyi anlayabilirler.

## Değerlendirme Kodları
Aşağıdaki kod örneği, bir üretken yapay zeka modelinin performansını değerlendirmek için kullanılan basit bir değerlendirme metriğini göstermektedir:
```python
import numpy as np
from sklearn.metrics import accuracy_score

# Modelin ürettiği çıktıları değerlendirmek için bir fonksiyon tanımlayalım
def evaluate_model_outputs(model_outputs, ground_truth):
    # Modelin ürettiği çıktıların doğruluğunu hesaplayalım
    accuracy = accuracy_score(ground_truth, model_outputs)
    return accuracy

# Örnek kullanım:
model_outputs = np.array([0, 1, 1, 0])
ground_truth = np.array([0, 1, 0, 0])

accuracy = evaluate_model_outputs(model_outputs, ground_truth)
print("Modelin doğruluğu:", accuracy)
```
## Kod Açıklaması
Yukarıdaki kodda, `evaluate_model_outputs` fonksiyonu, bir üretken yapay zeka modelinin ürettiği çıktıları (`model_outputs`) gerçek değerlerle (`ground_truth`) karşılaştırarak modelin doğruluğunu (`accuracy`) hesaplar. `accuracy_score` fonksiyonu, `sklearn.metrics` modülünden import edilir ve doğruluk skorunu hesaplamak için kullanılır.

## Önemli Noktalar
* Matematiksel metrikler, üretken yapay zeka modellerinin performansını değerlendirmek için yeterli değildir.
* İnsan değerlendirmesi, modelin gerçek dünyadaki performansını değerlendirmek için gereklidir.
* Değerlendirme metrikleri, modelin ürettiği çıktıların kalitesini ve çeşitliliğini ölçmede önemlidir.

## Eklemeler
Üretken yapay zeka modellerinin değerlendirilmesinde, çeşitlilik (diversity) ve novelty gibi metrikler de kullanılabilir. Bu metrikler, modelin ürettiği çıktıların yenilikçiliğini ve çeşitliliğini ölçmede yardımcı olabilir.

---

## Metrics (E1)

## Değerlendirme Metrikleri (E1)

Bir modelin performansı, herhangi bir yapay zeka sistemi gibi, kosinüs benzerliği (cosine similarity) gibi matematiksel metrikler olmadan değerlendirilemez. Bu metrikler, geri alınan verilerin ilgili ve doğru olmasını sağlar. Veri noktaları arasındaki ilişkileri ve alaka düzeyini nicelendirerek, modelin performansı ve güvenilirliği için sağlam bir temel sağlarlar.

## Önemli Noktalar
* Matematiksel metrikler, modelin performansını değerlendirmek için gereklidir.
* Kosinüs benzerliği (cosine similarity) gibi metrikler, veri noktaları arasındaki benzerliği ölçer.
* Metrikler, geri alınan verilerin ilgili ve doğru olmasını sağlar.

## Kullanılan Metrikler
Kosinüs benzerliği (cosine similarity), iki vektör arasındaki benzerliği ölçmek için kullanılan bir metriktir. Aşağıdaki Python kodu, kosinüs benzerliğini hesaplamak için kullanılabilir:
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# İki vektör tanımlayalım
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

# Kosinüs benzerliğini hesaplayalım
similarity = cosine_similarity([vector1], [vector2])

print(similarity)
```
## Kod Açıklaması
Yukarıdaki kodda, `cosine_similarity` fonksiyonu, iki vektör arasındaki kosinüs benzerliğini hesaplamak için kullanılır. Bu fonksiyon, `sklearn.metrics.pairwise` modülünden import edilir.

* `vector1` ve `vector2` adlı iki vektör tanımlanır.
* `cosine_similarity` fonksiyonu, bu iki vektör arasındaki kosinüs benzerliğini hesaplar.
* Sonuç, `similarity` değişkenine atanır ve yazdırılır.

## Kullanım Alanları
Kosinüs benzerliği, metin madenciliği (text mining), bilgi erişimi (information retrieval) ve öneri sistemleri (recommender systems) gibi birçok alanda kullanılır.

## Diğer Metrikler
Diğer bazı metrikler de modelin performansını değerlendirmek için kullanılabilir. Örneğin:
* Precision (Kesinlik)
* Recall (Geri çağırma)
* F1 Skoru
* Mean Average Precision (MAP)

Bu metrikler, modelin performansını farklı açılardan değerlendirmek için kullanılabilir.

---

## Human feedback (E2)

## İnsan Geri Bildirimi (E2)
Hiçbir üretken yapay zeka sistemi, RAG (Retrieval-Augmented Generation) tabanlı olsun ya da olmasın, matematiksel metriklerin yeterli görünse de görünmesin, insan değerlendirmesinden kaçamaz. Sonuç olarak, insan kullanıcılar için tasarlanan bir sistemin kabul edilip edilmeyeceğine, övülüp eleştirileceğine insan değerlendirmesi karar verir. Adaptif RAG, insan, gerçek hayat, pragmatik geri bildirim faktörünü tanıtır ve bu, RAG tabanlı üretken bir yapay zeka ekosistemini geliştirir.

## Önemli Noktalar:
- İnsan değerlendirmesi, bir yapay zeka sisteminin kabulü veya reddi konusunda nihai karara sahiptir (Human evaluation is the ultimate decider).
- RAG tabanlı veya değil, tüm üretken yapay zeka sistemleri insan değerlendirmesine tabidir (All generative AI systems are subject to human evaluation).
- Adaptif RAG, insan geri bildirim faktörünü içerir (Adaptive RAG introduces the human feedback factor).
- Adaptif RAG, RAG tabanlı üretken yapay zeka ekosistemini geliştirir (Adaptive RAG improves RAG-driven generative AI ecosystem).

## Kodlar ve Açıklamaları:
Paragrafta doğrudan bir kod verilmemektedir. Ancak, RAG ve Adaptif RAG konseptlerinin anlaşılması için ilgili Python kütüphanelerini kullanarak basit bir örnek kodlayabiliriz. Örneğin, `transformers` kütüphanesini kullanarak bir RAG modelini nasıl kullanabileceğimize dair bir örnek:

```python
## Gerekli Kütüphanelerin İçe Aktarılması (Importing Necessary Libraries)
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

## RAG Modelinin ve Tokenizer'ın Yüklenmesi (Loading RAG Model and Tokenizer)
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

## Giriş Metninin Hazırlanması (Preparing Input Text)
input_dict = tokenizer.prepare_seq2seq_batch("How many dogs are there?", return_tensors="pt")

## Çıktının Üretilmesi (Generating Output)
generated_ids = model.generate(input_ids=input_dict["input_ids"], attention_mask=input_dict["attention_mask"])

## Üretilen Çıktının Metne Dönüştürülmesi (Converting Generated IDs to Text)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

## Sonuç (Result)
print(generated_text)
```

## Kodun Açıklaması:
- `transformers` kütüphanesinden `RagTokenizer`, `RagRetriever`, ve `RagSequenceForGeneration` sınıflarını içe aktarıyoruz (`from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration`).
- RAG modelini ve tokenizer'ı yüklüyoruz (`tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")`, `retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)`, `model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)`).
- Giriş metnini hazırlıyoruz (`input_dict = tokenizer.prepare_seq2seq_batch("How many dogs are there?", return_tensors="pt")`).
- Modeli kullanarak çıktı üretiyoruz (`generated_ids = model.generate(input_ids=input_dict["input_ids"], attention_mask=input_dict["attention_mask"])`).
- Üretilen çıktı ID'lerini metne çeviriyoruz (`generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]`).
- Son olarak, üretilen metni yazdırıyoruz (`print(generated_text)`).

Bu örnek, RAG modelinin nasıl kullanılabileceğine dair basit bir gösterimdir. Adaptif RAG'ın nasıl uygulanacağı ise daha karmaşık bir konudur ve insan geri bildirim mekanizmalarının entegrasyonunu içerir. Bu konuya 5. Bölümde (`Chapter 5`) daha detaylı olarak girilecektir: `Boosting RAG Performance with Expert Human Feedback`.

---

## The trainer (T)

## Eğitmen (T)

Standart bir üretken yapay zeka modeli (Generative AI Model), geniş bir genel amaçlı veri seti ile önceden eğitilir (pre-trained). Daha sonra, model alan-specifik (domain-specific) verilerle fine-tune (T2) edilebilir. Biz bu süreci, 9. Bölüm'de statik RAG verilerini fine-tuning sürecine entegre ederek daha da ileri götüreceğiz. Ayrıca, insan geri bildirimlerini (human feedback) de entegre edeceğiz, bu değerli bilgiler Reinforcement Learning from Human Feedback (RLHF) varyantında fine-tuning sürecine dahil edilebilir.

## Önemli Noktalar
* Standart üretken yapay zeka modeli geniş bir genel amaçlı veri seti ile önceden eğitilir (pre-trained) (Generative AI Model Pre-training).
* Model alan-specifik verilerle fine-tune edilebilir (Domain-Specific Fine-Tuning).
* Statik RAG verileri fine-tuning sürecine entegre edilebilir (RAG Data Integration).
* İnsan geri bildirimleri Reinforcement Learning from Human Feedback (RLHF) varyantında fine-tuning sürecine dahil edilebilir (Human Feedback Integration).

## Python'da RAG Kodlama
Python'da entry-level naif, advanced ve modüler RAG kodlamak için hazırız.

## Kod Örneği
Aşağıdaki kod örneğinde, RAG modelini Python'da nasıl kodlayabileceğimizi göreceğiz.
```python
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Model ve tokenizer yükleme
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Veri yükleme ve ön işleme
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Veri ön işleme adımları
    pass

# RAG modeli eğitme
def train_rag_model(model, tokenizer, data):
    # Eğitim adımları
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # ...
    return model

# Kod kullanımı
data = load_data("data.csv")
data = preprocess_data(data)
model = train_rag_model(model, tokenizer, data)
```
## Kod Açıklaması
* `import` ifadeleri gerekli kütüphaneleri yükler.
* `AutoModelForSeq2SeqLM` ve `AutoTokenizer` sınıfları, önceden eğitilmiş T5 modelini ve tokenizer'ı yükler.
* `load_data` fonksiyonu, veri setini yükler.
* `preprocess_data` fonksiyonu, veri ön işleme adımlarını gerçekleştirir.
* `train_rag_model` fonksiyonu, RAG modelini eğitir.
* `device` değişkeni, eğitimin GPU veya CPU'da yapılacağını belirler.

## Notlar
* Kod örneğinde, T5 modeli kullanılmıştır, ancak başka modeller de kullanılabilir.
* Veri ön işleme adımları, veri setine göre değişebilir.
* Eğitim adımları, model ve veri setine göre değişebilir.

---

## Naïve, advanced, and modular RAG in code

## Naif, Gelişmiş ve Modüler RAG (Retrieval-Augmented Generation) Kodda Uygulanması
Bu bölümde, temel eğitim örnekleri aracılığıyla naif, gelişmiş ve modüler RAG (Retrieval-Augmented Generation) tanıtılmaktadır. Program, anahtar kelime eşleme (keyword matching), vektör arama (vector search) ve indeks tabanlı erişim (index-based retrieval) yöntemleri oluşturur. OpenAI'ın GPT modellerini kullanarak, girdi sorgularına (input queries) ve erişilen belgelere (retrieved documents) dayalı yanıtlar üretir. Notebook'un amacı, genel olarak RAG hakkında soruları yanıtlayabilen bir konuşma ajanına (conversational agent) sahip olmaktır.

## Temel Uygulama ve Altyapı (Foundations and Basic Implementation)
İlk olarak, OpenAI API entegrasyonu için ortamı (environment) kuracağız.

### Ortam Kurulumu (Environment Setup)
```python
import os
import openai

# OpenAI API anahtarını ayarlayın
openai.api_key = os.getenv("OPENAI_API_KEY")
```
Yukarıdaki kodda, OpenAI API anahtarını ortam değişkenlerinden (environment variables) alıyoruz. Bu, güvenlik ve esneklik açısından önemlidir.

### Jeneratör Fonksiyonu (Generator Function) GPT-4o ile
```python
import openai

def generate_response(prompt):
    response = openai.Completion.create(
        engine="gpt-4o",
        prompt=prompt,
        max_tokens=1024,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Örnek kullanım
prompt = "RAG nedir?"
print(generate_response(prompt))
```
Bu kod, GPT-4o modelini kullanarak verilen bir prompt'a dayalı metin üretir. `max_tokens` parametresi, üretilen metnin maksimum uzunluğunu belirler. `temperature` parametresi ise üretilen metnin yaratıcılığını kontrol eder.

### Veri Kurulumu (Data Setup)
```python
db_records = [
    {"id": 1, "text": "RAG, Retrieval-Augmented Generation anlamına gelir."},
    {"id": 2, "text": "RAG, bilgi erişimi ve metin üretimini birleştirir."},
    # Diğer belgeler...
]

# Kullanıcı girdisi için sorgu
query = input("Sorgunuzu girin: ")
```
Burada, `db_records` adlı bir liste içinde örnek belgelerimizi saklıyoruz. Kullanıcıdan girdi almak için `input` fonksiyonunu kullanıyoruz.

## Gelişmiş Teknikler ve Değerlendirme (Advanced Techniques and Evaluation)
Bu bölümde, erişim metrikleri (retrieval metrics) ile naif, gelişmiş ve modüler RAG uygulamalarını inceleyeceğiz.

### Naif RAG (Naïve RAG) Anahtar Kelime Arama ile
```python
def naive_rag(query, db_records):
    results = []
    for record in db_records:
        if query.lower() in record["text"].lower():
            results.append(record)
    return results

# Örnek kullanım
query = "RAG"
results = naive_rag(query, db_records)
for result in results:
    print(result["text"])
```
Bu kod, basit bir anahtar kelime eşleme yöntemi kullanarak belgeleri arar.

### Gelişmiş RAG (Advanced RAG) Vektör Arama ile
```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Vektör modeli yükleme
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

def vector_search(query, db_records):
    query_vector = model.encode(query)
    results = []
    for record in db_records:
        record_vector = model.encode(record["text"])
        similarity = np.dot(query_vector, record_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(record_vector))
        if similarity > 0.7:  # Benzerlik eşiği
            results.append(record)
    return results

# Örnek kullanım
query = "RAG"
results = vector_search(query, db_records)
for result in results:
    print(result["text"])
```
Bu kod, SentenceTransformer kütüphanesini kullanarak belgeleri vektör uzayında temsil eder ve benzerlik araması yapar.

### Modüler RAG (Modular RAG) Esnek Erişim Yöntemleri ile
```python
class ModularRAG:
    def __init__(self, db_records):
        self.db_records = db_records

    def search(self, query, method="keyword"):
        if method == "keyword":
            return naive_rag(query, self.db_records)
        elif method == "vector":
            return vector_search(query, self.db_records)
        else:
            raise ValueError("Geçersiz arama yöntemi")

# Örnek kullanım
rag = ModularRAG(db_records)
query = "RAG"
results_keyword = rag.search(query, method="keyword")
results_vector = rag.search(query, method="vector")

print("Anahtar Kelime Arama Sonuçları:")
for result in results_keyword:
    print(result["text"])

print("Vektör Arama Sonuçları:")
for result in results_vector:
    print(result["text"])
```
Bu kod, farklı erişim yöntemlerini modüler bir şekilde birleştiren bir RAG sınıfı tanımlar.

Tüm bu kod parçaları, RAG'ın farklı yönlerini göstermek için tasarlanmıştır. Her bir bölüm, RAG'ın belirli bir yönünü ele alır ve örnek kullanım senaryoları sağlar.

---

## Part 1: Foundations and basic implementation

## Bölüm 1: Temeller ve Temel Uygulama (Part 1: Foundations and Basic Implementation)

Bu bölümde, ortamı kuracağız (set up the environment), üretici (generator) için bir fonksiyon oluşturacağız, biçimlendirilmiş bir yanıtı yazdırmak için bir fonksiyon tanımlayacağız ve kullanıcı sorgusunu (user query) tanımlayacağız.

### Ortamı Kurma (Setting Up the Environment)

İlk adım, ortamı kurmaktır. Bunu yapmak için gerekli kütüphaneleri yüklememiz gerekir. Aşağıdaki kod, gerekli kütüphaneleri yüklemek için kullanılır:
```python
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
```
Bu kodda, `transformers` kütüphanesinden `AutoModelForCausalLM` ve `AutoTokenizer` sınıflarını içe aktarıyoruz (import). Bu sınıflar, sırasıyla üretici modelini ve tokenleştiriciyi (tokenizer) oluşturmak için kullanılır.

### Üretici Fonksiyonunu Tanımlama (Defining the Generator Function)

Üretici fonksiyonu, girdi olarak bir metin alan ve çıktı olarak yeni bir metin üreten bir fonksiyondur. Aşağıdaki kod, üretici fonksiyonunu tanımlar:
```python
def generate_text(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
```
Bu kodda, `generate_text` fonksiyonu, `model`, `tokenizer`, `prompt` ve `max_length` parametrelerini alır. `tokenizer` kullanarak `prompt` metnini tokenleştirir ve `model` kullanarak yeni metin üretir. Üretilen metin, `tokenizer.decode` yöntemi kullanılarak çözülür (decode) ve döndürülür.

### Biçimlendirilmiş Yanıtı Yazdırmak için Fonksiyon Tanımlama (Defining a Function to Print a Formatted Response)

Biçimlendirilmiş yanıtı yazdırmak için bir fonksiyon tanımlayacağız. Aşağıdaki kod, bu fonksiyonu tanımlar:
```python
def print_formatted_response(response):
    print("Yanıt:")
    print(response)
    print("-" * 50)
```
Bu kodda, `print_formatted_response` fonksiyonu, `response` parametresini alır ve biçimlendirilmiş bir şekilde yazdırır.

### Kullanıcı Sorgusunu Tanımlama (Defining the User Query)

Kullanıcı sorgusunu tanımlayacağız. Aşağıdaki kod, kullanıcı sorgusunu tanımlar:
```python
user_query = "Merhaba, nasılsınız?"
```
Bu kodda, `user_query` değişkenine bir metin atanır.

Tüm kodu bir araya getirdiğimizde:
```python
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def print_formatted_response(response):
    print("Yanıt:")
    print(response)
    print("-" * 50)

user_query = "Merhaba, nasılsınız?"

# Model ve tokenleştiriciyi oluşturma
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Üretici fonksiyonunu çağırma
generated_text = generate_text(model, tokenizer, user_query)

# Biçimlendirilmiş yanıtı yazdırma
print_formatted_response(generated_text)
```
Bu kod, üretici modelini kullanarak kullanıcı sorgusuna yanıt üretir ve biçimlendirilmiş bir şekilde yazdırır.

---

## 1. Environment

## 1. Ortam (Environment)
Bu bölümde, GPT-4o'ya (GPT-4o) API (Application Programming Interface) üzerinden erişmek için gerekli olan ana paketin kurulumu anlatılmaktadır.

### Önemli Noktalar:
- OpenAI (OpenAI) kütüphanesinin kurulumu
- Kütüphane versiyonunun dondurulması (Freezing the library version)
- OpenAI API (OpenAI API) anahtarının alınması ve güvenli bir şekilde saklanması
- API (API) anahtarının kod içinde kullanılması

### Kodlar ve Açıklamalar:

#### OpenAI Kütüphanesinin Kurulumu
```python
!pip install openai==1.40.3
```
Bu kod, OpenAI (OpenAI) kütüphanesini 1.40.3 versiyonu ile kurar. Versiyonun dondurulması, ileride oluşabilecek çakışmaları (conflicts) önlemek içindir.

#### Google Drive'ın Bağlanması
```python
from google.colab import drive
drive.mount('/content/drive')
```
Bu kod, Google Colab (Google Colab) içinde Google Drive'ı (Google Drive) bağlamak için kullanılır. API (API) anahtarını güvenli bir şekilde saklamak için kullanılır.

#### API Anahtarının Okunması ve Kullanılması
```python
f = open("drive/MyDrive/files/api_key.txt", "r")
API_KEY = f.readline().strip()
f.close()

import os
import openai
os.environ['OPENAI_API_KEY'] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
```
Bu kodlar, API (API) anahtarını `api_key.txt` dosyasından okur, ortam değişkeni (environment variable) olarak atar ve OpenAI (OpenAI) kütüphanesine bu anahtarı tanımlar.

- `f = open("drive/MyDrive/files/api_key.txt", "r")`: `api_key.txt` dosyasını okumak için açar.
- `API_KEY = f.readline().strip()`: Dosyanın ilk satırını okur ve gereksiz boşlukları temizler.
- `os.environ['OPENAI_API_KEY'] = API_KEY`: API (API) anahtarını ortam değişkeni olarak kaydeder.
- `openai.api_key = os.getenv("OPENAI_API_KEY")`: OpenAI (OpenAI) kütüphanesine API (API) anahtarını tanımlar.

Bu adımlarla, projenin ana kaynakları kurulmuş olur ve OpenAI model (OpenAI model) için bir üretim fonksiyonu yazmaya hazır olunur.

---

## 2. The generator

## Jeneratör (The Generator)

Jeneratör, içerik oluşturmak için OpenAI kütüphanesini (`openai`) ve isteklerin ne kadar sürdüğünü ölçmek için `time` kütüphanesini içe aktarır (`import`).

### Kod
```python
import openai
from openai import OpenAI
import time

client = OpenAI()
gpt_model = "gpt-4o"

start_time = time.time()  # İsteğin başlangıç zamanını kaydet
```

### Açıklama
- `openai` kütüphanesi, OpenAI API'sine erişim sağlar.
- `OpenAI` sınıfı, OpenAI API'sine bağlanmak için kullanılır.
- `time` kütüphanesi, zaman ölçümü için kullanılır.
- `client` değişkeni, OpenAI API'sine bağlanmak için kullanılan bir nesne (`object`) oluşturur.
- `gpt_model` değişkeni, kullanılacak GPT modelini belirtir (`"gpt-4o"`).
- `start_time` değişkeni, isteğin başlangıç zamanını kaydeder.

## Jeneratör Fonksiyonu (The Generator Function)

Jeneratör fonksiyonu, bir talimat (`instruction`) ve kullanıcı girdisini (`user input`) birleştirerek bir istem (`prompt`) oluşturur.

### Kod
```python
def call_llm_with_full_text(itext):
    text_input = '\n'.join(itext)
    prompt = f"Please elaborate on the following content:\n{text_input}"
    try:
        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": "You are an expert Natural Language Processing exercise expert."},
                {"role": "assistant", "content": "1.You can explain read the input and answer in detail"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Sıcaklık parametresi
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return str(e)
```

### Açıklama
- `call_llm_with_full_text` fonksiyonu, bir metin (`itext`) alır ve bir istem (`prompt`) oluşturur.
- `text_input` değişkeni, girdiyi (`itext`) tek bir metin haline getirir.
- `prompt` değişkeni, talimat ve girdiyi birleştirerek bir istem oluşturur.
- `client.chat.completions.create` metodu, OpenAI API'sine bir istek gönderir.
- `model` parametresi, kullanılacak GPT modelini belirtir (`gpt_model`).
- `messages` parametresi, sohbet geçmişini (`conversation history`) temsil eden bir liste (`list`) içerir.
- `temperature` parametresi, modelin yaratıcılığını (`creativity`) kontrol eder (`0.1`).

## Cevabı Biçimlendirme (Formatting the Response)

Cevabı biçimlendirmek için `textwrap` kütüphanesi kullanılır.

### Kod
```python
import textwrap

def print_formatted_response(response):
    wrapper = textwrap.TextWrapper(width=80)
    wrapped_text = wrapper.fill(text=response)
    print("Response:")
    print("---------------")
    print(wrapped_text)
    print("---------------\n")
```

### Açıklama
- `textwrap` kütüphanesi, metni biçimlendirmek için kullanılır.
- `print_formatted_response` fonksiyonu, bir cevabı (`response`) biçimlendirir.
- `wrapper` değişkeni, metni sarmak (`wrap`) için bir nesne (`object`) oluşturur.
- `wrapped_text` değişkeni, biçimlendirilmiş metni içerir.
- `print` fonksiyonları, biçimlendirilmiş cevabı yazdırır.

## Jeneratör Hazır (The Generator is Ready)

Jeneratör artık çağrılmaya hazırdır. Üretken (`generative`) AI modellerinin olasılıksal (`probabilistic`) doğası nedeniyle, her çağrıldığında farklı çıktılar üretebilir.

---

## 3. The Data

## Veri (The Data)
Veri toplama (data collection) metin, resim, ses ve video içerir. Bu notebook'ta, veri toplama yerine basit, gelişmiş ve modüler konfigürasyonlar aracılığıyla veri erişimi (data retrieval) üzerine odaklanacağız. Verileri daha sonra 2. Bölüm'de, "RAG Embedding Vector Stores with Deep Lake and OpenAI" başlığı altında toplayacak ve gömeceğiz (embedding). Bu nedenle, ihtiyacımız olan verilerin işlendiğini ve bir Python listesi olan `db_records` adlı bir değişkene yüklendiğini varsayacağız.

Bu yaklaşım, "The RAG Ecosystem" bölümünde açıklanan RAG ekosisteminin üç yönünü ve Şekil 1.3'te açıklanan sistem bileşenlerini gösterir:
- Erişim bileşeni (D) üç veri işleme (data processing) bileşenine sahiptir: toplama (D1), işleme (D2) ve depolama (D3), ki bunlar erişimin hazırlık aşamalarıdır.
- Erişim sorgusu (D4) bu nedenle erişimin ilk üç aşamasından (toplama, işleme ve depolama) bağımsızdır.
- Veri işleme aşaması genellikle erişim sorgusundan önce ve bağımsız olarak yapılır, ki biz de 2. Bölüm'den itibaren uygulayacağız.

## Veri Hazırlama
Bu program, veri işlemenin tamamlandığını ve veri setinin hazır olduğunu varsayar:
```python
db_records = [
    "Retrieval Augmented Generation (RAG) represents a sophisticated hybrid approach in the field of artificial intelligence, particularly within the realm of natural language processing (NLP).",
    "Retrieval Augmented Generation (RAG) represents a sophisticated hybrid approach in the field of artificial intelligence, particularly within the realm of natural language processing (NLP).",
    # ...
]
```
## Veri Gösterimi
Veri setinin biçimlendirilmiş bir sürümünü görüntüleyebiliriz:
```python
import textwrap

paragraph = ' '.join(db_records)
wrapped_text = textwrap.fill(paragraph, width=80)
print(wrapped_text)
```
Bu kod, `db_records` içindeki cümleleri birleştirir ve 80 karakter genişliğinde bir paragraf olarak biçimlendirir.

### Kod Açıklaması
- `import textwrap`: `textwrap` modülünü içe aktarır, ki bu metni biçimlendirmek için kullanılır.
- `paragraph = ' '.join(db_records)`: `db_records` listesindeki tüm cümleleri birleştirerek bir paragraf oluşturur.
- `wrapped_text = textwrap.fill(paragraph, width=80)`: Oluşturulan paragrafı 80 karakter genişliğinde biçimlendirir.
- `print(wrapped_text)`: Biçimlendirilmiş metni yazdırır.

Bu işlemler sonucunda, `db_records` değişmeden kalırken, birleştirilmiş ve biçimlendirilmiş metin görüntülenir:
```
Retrieval Augmented Generation (RAG) represents a sophisticated hybrid approach in the field of artificial intelligence, particularly within the realm of natural language processing (NLP)... 
Retrieval Augmented Generation (RAG) represents a sophisticated hybrid approach in the field of artificial intelligence, particularly within the realm of natural language processing (NLP)... 
Retrieval Augmented Generation (RAG) represents a sophisticated hybrid approach in the field of artificial intelligence, particularly within the realm of natural language processing (NLP)...
```
## Sorgu İşleme
Program artık bir sorguyu işlemeye hazırdır.

### Önemli Noktalar
- Veri toplama ve işleme aşamaları erişim sorgusundan bağımsızdır.
- Veri işleme genellikle erişim sorgusundan önce yapılır.
- `db_records` adlı Python listesi veri setini içerir.
- `textwrap` modülü metni biçimlendirmek için kullanılır.

---

## 4.The query

## Sorgu (The Query)
Bir organizasyondaki yüzlerce kullanıcı "RAG" terimini "LLM" ve "vector stores" ile ilişkilendirerek duymuştur. Birçoğu, departmanlarında bir konuşma aracı (conversational agent) dağıtan bir yazılım ekibiyle aynı hızda ilerlemek için bu terimlerin ne anlama geldiğini anlamak istiyor. Birkaç gün sonra, duydukları terimler hafızalarında bulanıklaşıyor ve konuşma aracına (bu durumda GPT-4o) hatırladıklarını açıklamasını istiyorlar: 
## Sorgu Örneği
```python
query = "define a rag store"
```
Bu sorgu, retriever ( D4 ) ve generator arasındaki bağlantıyı temsil eder ve bir RAG konfigürasyonunu (naïve, advanced ve modular) tetikler. Projenin hedeflerine bağlı olarak konfigürasyon seçimi yapılır.

## Sorgunun İşlenmesi
Program, sorguyu alır ve bir GPT-4o modeline göndererek işler ve daha sonra biçimlendirilmiş çıktıyı görüntüler:
```python
llm_response = call_llm_with_full_text(query)
print_formatted_response(llm_response)
```
## Çıktı (Output)
GPT-4o'nun çıktısı şöyledir:
```
Response:
---------------
Certainly! The content you've provided appears to be a sequence of characters
that, when combined, form the phrase "define a rag store." Let's break it down
step by step:…
… This is an indefinite article used before words that begin with a consonant sound.
    - **rag**: This is a noun that typically refers to a piece of old, often torn, cloth.
    - **store**: This is a noun that refers to a place where goods are sold.
  4. **Contextual Meaning**:
    - **"Define a rag store"**: This phrase is asking for an explanation or definition of what a "rag store" is.
  5. **Possible Definition**:
    - A "rag store" could be a shop or retail establishment that specializes in selling rags,…
…
Would you like more information or a different type of elaboration on this content?…
```
GPT-4o, olasılıksal algoritması (probabilistic algorithm) ile sınırlı bağlamda yapabileceği en iyi şeyi yapar, ancak sorgu net değildir, bu nedenle daha fazla içerik ister.

## Önemli Noktalar
* Sorgu, retriever ve generator arasındaki bağlantıyı temsil eder.
* RAG konfigürasyonu (naïve, advanced ve modular) sorgu tarafından tetiklenir.
* GPT-4o, olasılıksal algoritması ile sınırlı bağlamda yapabileceği en iyi şeyi yapar.
* Sorgu net değilse, GPT-4o daha fazla içerik ister.

## Kullanılan Kodlar ve Açıklamaları
### `query` Değişkeni
```python
query = "define a rag store"
```
Bu kod, sorguyu `query` değişkenine atar.

### `call_llm_with_full_text` Fonksiyonu
```python
llm_response = call_llm_with_full_text(query)
```
Bu kod, `query` değişkenini `call_llm_with_full_text` fonksiyonuna gönderir ve GPT-4o modelinin çıktısını `llm_response` değişkenine atar.

### `print_formatted_response` Fonksiyonu
```python
print_formatted_response(llm_response)
```
Bu kod, GPT-4o modelinin çıktısını biçimlendirerek görüntüler.

### Import Kısımları
Bu kodda import kısımları gösterilmemiştir, ancak `call_llm_with_full_text` ve `print_formatted_response` fonksiyonlarını çağırmak için gerekli olan import ifadeleri eklenmelidir. Örneğin:
```python
import necessary_library
```
Bu import ifadeleri, kullanılan kütüphanelere bağlı olarak değişebilir.

## Sonuç
RAG, belirsiz sorguları ele almada ve daha doğru sonuçlar üretmede yardımcı olur. Bu örnekte, GPT-4o'nun olasılıksal algoritması ile sınırlı bağlamda yapabileceği en iyi şeyi yaptığı görülmüştür. Ancak, RAG konfigürasyonu ile daha iyi sonuçlar elde edilebilir.

---

## Part 2: Advanced techniques and evaluation

## Bölüm 2: Gelişmiş Teknikler ve Değerlendirme

## Giriş
Bu bölümde, basit, gelişmiş ve modüler RAG (Retrieval-Augmented Generation) tekniklerini tanıtacağız. Amacımız, bu üç yöntemi tanıtmak ve ilerleyen bölümlerde daha karmaşık belgeleri işleme koymaktır. İlk olarak, geri çağırdığımız belgelerin doğruluğunu ölçmek için geri çağırma metriklerini tanımlayarak başlayalım.

## Geri Çağırma Metrikleri (Retrieval Metrics)
Geri çağırma metrikleri, geri çağırdığımız belgelerin doğruluğunu ölçmek için kullanılır. Bu metrikler, `doğruluk (accuracy)`, `kesinlik (precision)`, `geri çağırma (recall)` ve `F1 skoru` gibi ölçütleri içerir.

### Doğruluk (Accuracy)
Doğruluk, geri çağırdığımız belgelerin ne kadarının ilgili olduğu ölçüsüdür.

### Kesinlik (Precision)
Kesinlik, geri çağırdığımız belgelerin ne kadarının gerçekten ilgili olduğu ölçüsüdür.

### Geri Çağırma (Recall)
Geri çağırma, ilgili belgelerin ne kadarının geri çağırıldığı ölçüsüdür.

### F1 Skoru
F1 skoru, kesinlik ve geri çağırma arasında bir denge kurar.

## Değerlendirme Kodları
Aşağıdaki kod, geri çağırma metriklerini hesaplamak için kullanılır:
```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Örnek veri
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 0])

# Doğruluk (Accuracy)
accuracy = accuracy_score(y_true, y_pred)
print("Doğruluk (Accuracy):", accuracy)

# Kesinlik (Precision)
precision = precision_score(y_true, y_pred)
print("Kesinlik (Precision):", precision)

# Geri Çağırma (Recall)
recall = recall_score(y_true, y_pred)
print("Geri Çağırma (Recall):", recall)

# F1 Skoru
f1 = f1_score(y_true, y_pred)
print("F1 Skoru:", f1)
```
Bu kod, `y_true` ve `y_pred` dizilerini kullanarak doğruluk, kesinlik, geri çağırma ve F1 skoru hesaplar.

## Kod Açıklaması
*   `y_true` ve `y_pred` dizileri, gerçek etiketler ve tahmin edilen etiketleri temsil eder.
*   `accuracy_score`, `precision_score`, `recall_score` ve `f1_score` fonksiyonları, sırasıyla doğruluk, kesinlik, geri çağırma ve F1 skoru hesaplar.
*   Bu fonksiyonlar, `sklearn.metrics` modülünden içe aktarılır.

## RAG Teknikleri
Bu bölümde, basit, gelişmiş ve modüler RAG tekniklerini tanıtacağız.

### Basit RAG (Naïve RAG)
Basit RAG, belgeleri geri çağırmak için basit bir yaklaşım kullanır.

### Gelişmiş RAG (Advanced RAG)
Gelişmiş RAG, belgeleri geri çağırmak için daha karmaşık bir yaklaşım kullanır.

### Modüler RAG (Modular RAG)
Modüler RAG, belgeleri geri çağırmak için modüler bir yaklaşım kullanır.

## Sonuç
Bu bölümde, geri çağırma metriklerini tanımladık ve RAG tekniklerini tanıttık. İlerleyen bölümlerde, daha karmaşık belgeleri işleme koymak için bu teknikleri kullanacağız.

---

## 1. Retrieval metrics

## 1. Bilgi Erişim Metrikleri (Retrieval Metrics)

Bu bölümde, metin belgelerinin (text documents) alaka düzeyini değerlendirmede kullanılan cosine similarity'nin (kosinüs benzerliği) rolüne odaklanacağız. Ardından, metinler arasındaki benzerlik hesaplamalarının doğruluğunu artırmak için sinonim genişletmesi (synonym expansion) ve metin ön işleme (text preprocessing) tekniklerini kullanarak gelişmiş benzerlik metriklerini (enhanced similarity metrics) uygulayacağız.

### Kosinüs Benzerliği (Cosine Similarity)

Kosinüs benzerliği, iki vektör arasındaki açının kosinüsünü ölçer. Bizim durumumuzda, iki vektör kullanıcı sorgusu (user query) ve bir derlemdeki (corpus) her bir belgedir (document).

#### Kosinüs Benzerliği Hesaplamak için Gerekli Kod

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        use_idf=True,
        norm='l2',
        ngram_range=(1, 2),  # Use unigrams and bigrams
        sublinear_tf=True,  # Apply sublinear TF scaling
        analyzer='word'  # You could also experiment with 'char' or 'char_wb' for character-level features
    )
    tfidf = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]
```

Bu fonksiyonun önemli parametreleri:

*   `stop_words='english'`: İngilizce yaygın kelimeleri yok sayar, anlamlı içeriğe odaklanmak için.
*   `use_idf=True`: Ters belge frekansı ağırlıklandırmasını etkinleştirir.
*   `norm='l2'`: Her çıktı vektörüne L2 normalizasyonu uygular.
*   `ngram_range=(1, 2)`: Hem tek kelimeleri hem de iki kelimelik kombinasyonları dikkate alır.
*   `sublinear_tf=True`: Logaritmik terim frekansı ölçeklendirmesini uygular.
*   `analyzer='word'`: Metni kelime düzeyinde analiz eder.

#### Kosinüs Benzerliğinin Sınırlamaları

Kosinüs benzerliği, belirsiz sorgularla (ambiguous queries) başa çıkmada sınırlamalara sahiptir çünkü vektör temsilleri arasındaki açıya dayalı benzerliği kesin olarak ölçer. Kullanıcı "RAG nedir?" gibi belirsiz bir soru sorduğunda ve veritabanı öncelikle "RAG" ile ilgili bilgileri "retrieval-augmented generation" için içerdiğinde, kosinüs benzerliği skoru düşük olabilir.

### Gelişmiş Benzerlik (Enhanced Similarity)

Gelişmiş benzerlik, kelimeler arasındaki anlamsal ilişkileri daha iyi yakalamak için doğal dil işleme araçlarını kullanan hesaplamaları tanıtır. spaCy ve NLTK gibi kütüphaneleri kullanarak, metinleri gürültüyü azaltmak için ön işleme tabi tutar, WordNet'ten sinonimlerle terimleri genişletir ve genişletilmiş kelime hazinesinin anlamsal zenginliğine dayalı benzerliği hesaplar.

#### Gelişmiş Benzerlik Hesaplamak için Gerekli Kod

```python
import spacy
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from collections import Counter
import numpy as np

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def get_synonyms(word):
    # Verilen bir kelimenin sinonimlerini WordNet'ten alır.
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def preprocess_text(text):
    # Tüm metni küçük harfe çevirir, kelimelerin köklerini alır ve yaygın kelimeleri ve noktalama işaretlerini filtreler.
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens

def expand_with_synonyms(words):
    # Bir kelime listesini sinonimleriyle genişletir.
    expanded_words = []
    for word in words:
        synonyms = get_synonyms(word)
        expanded_words.extend(synonyms)
    return expanded_words

def calculate_enhanced_similarity(text1, text2):
    # Ön işleme tabi tutulmuş ve sinonimlerle genişletilmiş metin vektörleri arasındaki kosinüs benzerliğini hesaplar.
    tokens1 = preprocess_text(text1)
    tokens2 = preprocess_text(text2)
    expanded_tokens1 = expand_with_synonyms(tokens1)
    expanded_tokens2 = expand_with_synonyms(tokens2)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([' '.join(expanded_tokens1), ' '.join(expanded_tokens2)])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]
```

### Gelişmiş Benzerliğin Avantajları

Gelişmiş benzerlik, metinler arasındaki benzerliği daha geniş bir bağlamda dikkate alarak daha doğru bir şekilde değerlendirir.

### RAG ile İlgili Sınırlamalar ve Zorluklar

RAG'ı üretken yapay zeka ile entegre etmek birçok zorluk sunar. Uyguladığımız metrik ne olursa olsun, aşağıdaki sınırlamalarla karşılaşacağız:

*   **Giriş ve Belge Uzunluğu**: Kullanıcı sorguları genellikle kısa, alınan belgeler ise daha uzun ve zengindir, bu da doğrudan benzerlik değerlendirmelerini zorlaştırır.
*   **Yaratıcı Erişim**: Sistemler, kullanıcı beklentilerini karşılayan ancak beklenmedik içerik hizalaması nedeniyle zayıf metrik skorları veren daha uzun belgeleri yaratıcı bir şekilde seçebilir.
*   **İnsan Geri Bildirimi İhtiyacı**: Otomatik metrikler kullanıcı memnuniyetini tam olarak yakalayamayabileceğinden, alınan içeriğin alaka düzeyini ve etkinliğini doğru bir şekilde değerlendirmek için genellikle insan yargısı çok önemlidir.

Bu zorlukların üstesinden gelmek için matematiksel metrikler ve insan geri bildirimi arasında doğru dengeyi bulmak her zaman önemlidir.

---

## 2. Naïve RAG

## Naïve RAG (Basit RAG) Konusu

Naïve RAG, belirli bir organizasyon içinde iyi tanımlanmış belgelerle, örneğin yasal ve tıbbi belgelerle verimli bir şekilde çalışabilir. Bu belgelerin genellikle resimler için net başlıkları veya etiketleri vardır. Bu Naïve RAG işlevinde, anahtar kelime araması ve eşleştirmesi uygulanacaktır.

### Naïve RAG İşlevi

1. Kullanıcı sorgusunu tek tek anahtar kelimelere ayır (`query_keywords = set(query.lower().split())`)
2. Veri kümesindeki her bir kaydı anahtar kelimelere ayır (`record_keywords = set(record.lower().split())`)
3. Ortak anahtar kelimelerin sayısını belirle (`common_keywords = query_keywords.intersection(record_keywords)`)
4. En iyi eşleşen kaydı seç (`best_record`)

### Kod

```python
def find_best_match_keyword_search(query, db_records):
    best_score = 0
    best_record = None

    # Sorguyu anahtar kelimelere ayır
    query_keywords = set(query.lower().split())

    # Veri kümesindeki her bir kaydı anahtar kelimelere ayır
    for record in db_records:
        record_keywords = set(record.lower().split())

        # Ortak anahtar kelimelerin sayısını belirle
        common_keywords = query_keywords.intersection(record_keywords)
        current_score = len(common_keywords)

        # Mevcut skor daha yüksekse en iyi skoru ve kaydı güncelle
        if current_score > best_score:
            best_score = current_score
            best_record = record

    return best_score, best_record

# Örnek kullanım
query = "define a rag store"
db_records = [...]  # Veri kümesi

best_keyword_score, best_matching_record = find_best_match_keyword_search(query, db_records)
print(f"En İyi Anahtar Kelime Skoru: {best_keyword_score}")
print_formatted_response(best_matching_record)
```

### Anahtar Kelime Arama ve Eşleştirme

Bu işlev, en iyi eşleşen kaydı bulmak için anahtar kelime araması ve eşleştirmesi yapar.

### Metrikler

1. Cosine Similarity (Kosinüs Benzerliği) (`calculate_cosine_similarity(query, best_matching_record)`)
2. Enhanced Similarity (Gelişmiş Benzerlik) (`calculate_enhanced_similarity(query, response)`)

### Kod

```python
# Kosinüs Benzerliği
score = calculate_cosine_similarity(query, best_matching_record)
print(f"En İyi Kosinüs Benzerliği Skoru: {score:.3f}")

# Gelişmiş Benzerlik
response = best_matching_record
similarity_score = calculate_enhanced_similarity(query, response)
print(f"Gelişmiş Benzerlik: {similarity_score:.3f}")
```

### Artırılmış Giriş (Augmented Input)

Kullanıcı girişi ve en iyi eşleşen kayıt birleştirilir (`augmented_input = query + ": " + best_matching_record`).

### Kod

```python
augmented_input = query + ": " + best_matching_record
print_formatted_response(augmented_input)
```

### Üretim (Generation)

Artırılmış giriş GPT-4o modeline gönderilir ve yanıt alınır (`llm_response = call_llm_with_full_text(augmented_input)`).

### Kod

```python
llm_response = call_llm_with_full_text(augmented_input)
print_formatted_response(llm_response)
```

Naïve RAG, birçok durumda yeterli olabilir. Ancak, belge hacmi çok büyük olduğunda veya içerik daha karmaşık hale geldiğinde, gelişmiş RAG konfigürasyonları daha iyi sonuçlar sağlayacaktır.

---

## 3. Advanced RAG

## Gelişmiş RAG (Advanced RAG)

Veri setleri büyüdükçe, anahtar kelime arama (keyword search) yöntemleri çok uzun sürmeye başlayabilir. Örneğin, yüzlerce belgeye ve her belgenin yüzlerce cümle içerdiği bir durumda, yalnızca anahtar kelime araması kullanmak zorlaşacaktır. Bir dizin (index) kullanmak, hesaplama yükünü toplam verinin yalnızca bir kısmına indirgeyecektir.

Bu bölümde, metni anahtar kelimelerle aramanın ötesine geçeceğiz. RAG'ın metin verilerini sayısal temsillere (numerical representations) nasıl dönüştürdüğünü, arama verimliliğini ve işleme hızını nasıl artırdığını göreceğiz. Geleneksel yöntemlerin doğrudan metni ayrıştırmasının aksine, RAG önce belgeleri ve kullanıcı sorgularını vektörlere (vector) dönüştürür, bu da hesaplamaları hızlandırır.

### Vektör Arama (Vector Search)

Vektör arama, kullanıcı sorgusunu ve belgeleri sayısal vektörler olarak dönüştürür, böylece büyük hacimli verilerle uğraşırken alakalı verileri daha hızlı almak için matematiksel hesaplamalar yapar. Program, veri setindeki her bir kaydı, sorgu vektörü (kullanıcı sorgusu) ve veri setindeki her bir kayıt arasındaki kosinüs benzerliğini (cosine similarity) hesaplayarak en iyi eşleşen belgeyi bulur.

#### Kod
```python
def find_best_match(text_input, records):
    best_score = 0
    best_record = None
    for record in records:
        current_score = calculate_cosine_similarity(text_input, record)
        if current_score > best_score:
            best_score = current_score
            best_record = record
    return best_score, best_record
```
Bu fonksiyon, `text_input` (kullanıcı sorgusu) ve `records` (belgeler) arasındaki kosinüs benzerliğini hesaplar ve en yüksek benzerlik puanına sahip belgeyi döndürür.

### Vektör Arama Sonuçları

Vektör arama fonksiyonu çağrıldığında, en iyi eşleşen kayıt görüntülenir:
```python
best_similarity_score, best_matching_record = find_best_match(query, db_records)
print_formatted_response(best_matching_record)
```
Çıktı:
```
Response:
---------------
A RAG vector store is a database or dataset that contains vectorized data points.
```
Vektör aramanın avantajı, büyük veri setlerinde daha hızlı sonuçlar vermesidir. Ancak, veri seti büyüdükçe, vektör aramanın verimliliği düşebilir.

### Dizin Tabanlı Arama (Index-Based Search)

Dizin tabanlı arama, kullanıcı sorgusunun vektörünü doğrudan belgenin içeriğinin vektörüyle değil, bu içeriği temsil eden dizinlenmiş bir vektörle karşılaştırır.

#### Kod
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_best_match(query, vectorizer, tfidf_matrix):
    query_tfidf = vectorizer.transform([query])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix)
    best_index = similarities.argmax()
    best_score = similarities[0, best_index]
    return best_score, best_index
```
Bu fonksiyon, `query` (kullanıcı sorgusu) ve `tfidf_matrix` (belgelerin TF-IDF vektörleri) arasındaki kosinüs benzerliğini hesaplar ve en yüksek benzerlik puanına sahip belgenin dizinini döndürür.

### Dizin Tabanlı Arama Sonuçları

Dizin tabanlı arama fonksiyonu çağrıldığında, en iyi eşleşen kayıt görüntülenir:
```python
vectorizer, tfidf_matrix = setup_vectorizer(db_records)
best_similarity_score, best_index = find_best_match(query, vectorizer, tfidf_matrix)
best_matching_record = db_records[best_index]
print_formatted_response(best_matching_record)
```
Çıktı:
```
Response:
---------------
A RAG vector store is a database or dataset that contains vectorized data points.
```
Dizin tabanlı aramanın avantajı, daha hızlı ve verimli olmasıdır.

### Artırılmış Giriş (Augmented Input)

Hem vektör arama hem de dizin tabanlı arama için, kullanıcı sorgusuna en iyi eşleşen belge eklenerek artırılmış bir giriş oluşturulur:
```python
augmented_input = query + ": " + best_matching_record
print_formatted_response(augmented_input)
```
Çıktı:
```
Response:
---------------
define a rag store: A RAG vector store is a database or dataset that contains vectorized data points.
```
### Üretken Yapay Zeka Modeli (Generative AI Model)

Artırılmış giriş, üretken yapay zeka modeline (GPT-4o) beslenir ve çıktı görüntülenir:
```python
llm_response = call_llm_with_full_text(augmented_input)
print_formatted_response(llm_response)
```
Çıktı:
```
Response:
---------------
Certainly! Let's break down and elaborate on the given content:  ---  **Define a RAG store:**  A **RAG vector store** is a **database** or **dataset** that contains **vectorized data points**.  ---  ### Detailed Explanation:  1. **RAG Store**:    - **RAG** stands for **Retrieval-Augmented Generation**. It is a technique used in natural language processing (NLP) where a model retrieves relevant information from a database or dataset to augment its generation capabilities…
```
Bu yaklaşım, kapalı bir ortamda iyi sonuçlar vermiştir. Ancak, açık bir ortamda, kullanıcının sorgusunu göndermeden önce detaylandırması gerekebilir.

Bu bölümde, TF-IDF matrisinin belge vektörlerini önceden hesapladığını ve daha hızlı karşılaştırmalar yapılmasını sağladığını gördük. Vektör ve dizin tabanlı aramanın, alma işlemini nasıl iyileştirebileceğini gördük. Ancak, bir projede, naive ve gelişmiş RAG'ı aynı anda uygulamak gerekebilir. Şimdi, modüler RAG'ın sistemimizi nasıl iyileştirebileceğini göreceğiz.

---

## 4. Modular RAG

## Modüler RAG (Modular RAG)

RAG (Retrieval-Augmented Generation) sistemlerinde hangi arama yönteminin kullanılacağına karar vermek önemlidir. Her bir arama yönteminin (anahtar kelime arama (keyword search), vektör arama (vector search), indeks tabanlı arama (index-based search)) kendine özgü avantajları vardır ve seçim çeşitli faktörlere bağlıdır.

### Arama Yöntemleri

*   **Anahtar Kelime Arama (Keyword Search)**: Basit veri alma (retrieval) işlemleri için uygundur.
*   **Vektör Arama (Vector Search)**: Anlam bakımından zengin belgeler (semantic-rich documents) için idealdir.
*   **İndeks Tabanlı Arama (Index-Based Search)**: Büyük veri setlerinde hızlı arama sağlar.

Bu üç yöntem de bir projede birlikte mükemmel bir şekilde kullanılabilir.

### RetrievalComponent Sınıfı

`RetrievalComponent` sınıfı, bir projenin her adımında gerekli görevi gerçekleştirmek için çağrılabilir. Bu sınıf, daha önce oluşturulan üç arama yöntemini bir araya getirir.

#### Kod

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RetrievalComponent:
    def __init__(self, method='vector'):
        self.method = method
        if self.method == 'vector' or self.method == 'indexed':
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = None

    def fit(self, records):
        if self.method == 'vector' or self.method == 'indexed':
            self.tfidf_matrix = self.vectorizer.fit_transform(records)

    def retrieve(self, query):
        if self.method == 'keyword':
            return self.keyword_search(query)
        elif self.method == 'vector':
            return self.vector_search(query)
        elif self.method == 'indexed':
            return self.indexed_search(query)

    def keyword_search(self, query):
        best_score = 0
        best_record = None
        query_keywords = set(query.lower().split())
        for index, doc in enumerate(self.documents):
            doc_keywords = set(doc.lower().split())
            common_keywords = query_keywords.intersection(doc_keywords)
            score = len(common_keywords)
            if score > best_score:
                best_score = score
                best_record = self.documents[index]
        return best_record

    def vector_search(self, query):
        query_tfidf = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)
        best_index = similarities.argmax()
        return db_records[best_index]

    def indexed_search(self, query):
        # tfidf_matrix önceden hesaplanmış ve saklanmış varsayılır
        query_tfidf = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)
        best_index = similarities.argmax()
        return db_records[best_index]
```

#### Açıklama

*   `__init__` methodu, sınıfın ilk örneğini oluştururken arama yöntemini seçmeye ve gerekirse bir vektörleştirici (vectorizer) hazırlamaya yarar.
*   `fit` methodu, TF-IDF matrisini oluşturur ve vektör veya indeks tabanlı arama yöntemleri için kullanılır.
*   `retrieve` methodu, sorguyu uygun arama yöntemine yönlendirir.
*   `keyword_search`, `vector_search` ve `indexed_search` methodları, sırasıyla anahtar kelime, vektör ve indeks tabanlı arama işlemlerini gerçekleştirir.

### Kullanım Örneği

```python
retrieval = RetrievalComponent(method='vector')  # 'keyword', 'vector', 'indexed' arasından seçim yapabilirsiniz
retrieval.fit(db_records)
best_matching_record = retrieval.retrieve(query)
print_formatted_response(best_matching_record)
```

Bu örnekte, vektör arama yöntemi etkinleştirilmiştir. `RetrievalComponent` sınıfı, farklı arama yöntemlerini bir araya getirerek esnek bir RAG sistemi oluşturmaya olanak tanır.

### Önemli Noktalar

*   Farklı arama yöntemleri (anahtar kelime, vektör, indeks tabanlı) çeşitli ihtiyaçları karşılar.
*   `RetrievalComponent` sınıfı, bu arama yöntemlerini birleştirir ve esnek bir RAG sistemi sağlar.
*   Arama yöntemi seçimi, veri seti boyutu, sorgu tipi ve performans gereksinimlerine bağlıdır.

---

## Summary

## RAG (Retrieval-Augmented Generation) Çerçevesi Özeti
RAG, üretken yapay zeka (Generative AI) için iki ana bileşene dayanır: bir retriever (bulucu) ve bir generator (üretici). Retriever, verileri işler ve anahtar kelimelerle etiketlenmiş belgeleri getirmek gibi bir arama yöntemi tanımlar. Generator'un girdisi olan bir LLM (Large Language Model), diziler üretirken artırılmış bilgiden yararlanır.

## RAG Çerçevesinin Ana Konfigürasyonları
RAG çerçevesinin üç ana konfigürasyonu vardır:
*   **Naive RAG**: Anahtar kelimeler ve diğer giriş seviyesindeki arama yöntemleri aracılığıyla veri setlerine erişir.
*   **Advanced RAG**: Arama yöntemlerini geliştirmek için embedding'leri (gömme) ve indeksleri tanıtır.
*   **Modular RAG**: Naive ve advanced RAG'ı diğer makine öğrenimi (ML) yöntemleriyle birleştirebilir.

## RAG Çerçevesinin Özellikleri
RAG çerçevesi, dinamik veriler içerebilen veri setlerine dayanır. Bir üretken yapay zeka modeli, ağırlıkları aracılığıyla parametrik verilere dayanır. Bu iki yaklaşım birbirini dışlamaz. RAG veri setleri çok hantal hale gelirse, ince ayar (fine-tuning) yararlı olabilir. İnce ayarlı modeller günlük bilgilere yanıt veremediğinde, RAG işe yarayabilir.

## RAG Ekosistemi
RAG çerçeveleri, sistemlerin çalışmasını sağlayan kritik işlevselliği sağlayan ekosisteme büyük ölçüde güvenmektedir. Retriever'dan generator'a kadar RAG ekosisteminin ana bileşenlerini inceledik. Eğitici (trainer) ve değerlendirici (evaluator) de gereklidir.

## Python'da RAG Programı Oluşturma
Son olarak, Python'da giriş seviyesinde bir naive, advanced ve modular RAG programı oluşturduk. Anahtar kelime eşleştirme (keyword matching), vektör arama (vector search) ve indeks tabanlı alma (index-based retrieval) kullanarak GPT-4o'nun girdisini artırdık.

## Örnek Kod
Aşağıdaki kod, RAG çerçevesini kullanarak bir üretken yapay zeka modeli oluşturmaya örnektir:
```python
# Import gerekli kütüphaneler
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Veri setini yükle
data = pd.read_csv("data.csv")

# Retriever oluştur
class Retriever:
    def __init__(self, data):
        self.data = data
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embedding(self, text):
        return self.model.encode(text)

    def search(self, query):
        query_embedding = self.get_embedding(query)
        data_embeddings = self.data["text"].apply(self.get_embedding)
        similarities = cosine_similarity([query_embedding], data_embeddings.tolist()).flatten()
        return self.data.iloc[similarities.argsort()[::-1]]

# Generator oluştur
class Generator:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, input_text):
        # GPT-4o modelini kullanarak metin üret
        import openai
        openai.api_key = "YOUR_API_KEY"
        response = openai.Completion.create(
            engine=self.model_name,
            prompt=input_text,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].text.strip()

# RAG modeli oluştur
class RAG:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def generate(self, query):
        retrieved_data = self.retriever.search(query)
        input_text = retrieved_data["text"].iloc[0]
        return self.generator.generate(input_text)

# Kullanım örneği
retriever = Retriever(data)
generator = Generator("GPT-4o")
rag = RAG(retriever, generator)
print(rag.generate("örnek sorgu"))
```
## Kod Açıklaması
Yukarıdaki kod, RAG çerçevesini kullanarak bir üretken yapay zeka modeli oluşturur. Retriever sınıfı, veri setini yükler ve anahtar kelime eşleştirme veya vektör arama kullanarak sorguları işler. Generator sınıfı, GPT-4o modelini kullanarak metin üretir. RAG sınıfı, retriever ve generator'ı birleştirerek RAG modelini oluşturur.

## Kullanılan Kütüphaneler ve Açıklamaları
*   `pandas`: Veri setini yüklemek ve işlemek için kullanılır.
*   `sentence-transformers`: Metinleri embedding'lemek için kullanılır.
*   `sklearn`: Kosinüs benzerliği hesaplamak için kullanılır.
*   `openai`: GPT-4o modelini kullanarak metin üretmek için kullanılır.

## İleri Adımlar
Bir sonraki adım, Chapter 2'de, RAG Embedding Vector Stores with Deep Lake ve OpenAI konularını ele almaktır. Veri vektörlerini depolamak için vektör depoları (vector stores) kullanılacaktır. Bu, RAG ekosisteminin hızını ve kesinliğini artıracaktır.

---

## Questions

## RAG (Retrieval-Augmented Generation) Hakkında Sorular ve Cevapları
RAG, üretken yapay zeka modellerinin (Generative AI Models) doğruluğunu artırmak için tasarlanmıştır.

### Sorular ve Cevaplar
1. ## RAG, üretken yapay zeka modellerinin doğruluğunu artırmak için mi tasarlanmıştır? (Is RAG designed to improve the accuracy of generative AI models?)
   - ## Cevap: Evet (Yes)

2. ## Naif RAG konfigürasyonu karmaşık veri embedding'lerine (Complex Data Embedding) dayanır mı? (Does a naïve RAG configuration rely on complex data embedding?)
   - ## Cevap: Hayır (No)

3. ## İnce ayar (Fine-Tuning) her zaman RAG'den daha iyi bir seçenek midir? (Is fine-tuning always a better option than using RAG?)
   - ## Cevap: Hayır (No)

4. ## RAG, yanıtları geliştirmek için gerçek zamanlı olarak harici kaynaklardan veri alır mı? (Does RAG retrieve data from external sources in real time to enhance responses?)
   - ## Cevap: Evet (Yes)

5. ## RAG yalnızca metin tabanlı verilere uygulanabilir mi? (Can RAG be applied only to text-based data?)
   - ## Cevap: Hayır (No)

6. ## RAG'deki retrieval işlemi kullanıcı tarafından tetiklenir mi yoksa otomatik mi? (Is the retrieval process in RAG triggered by a user or automated input?)
   - ## Cevap: Otomatik (Automated)

7. ## Kosinüs benzerliği (Cosine Similarity) ve TF-IDF, gelişmiş RAG konfigürasyonlarında kullanılan metrikler midir? (Are cosine similarity and TF-IDF both metrics used in advanced RAG configurations?)
   - ## Cevap: Evet (Yes)

8. ## RAG ekosistemi yalnızca veri toplama ve oluşturma bileşenlerini içerir mi? (Does the RAG ecosystem include only data collection and generation components?)
   - ## Cevap: Hayır (No)

9. ## Gelişmiş RAG konfigürasyonları, görüntüler ve ses gibi çok modlu verileri işleyebilir mi? (Can advanced RAG configurations process multimodal data such as images and audio?)
   - ## Cevap: Evet (Yes)

10. ## İnsan geri bildirimi (Human Feedback), RAG sistemlerini değerlendirmede önemsiz midir? (Is human feedback irrelevant in evaluating RAG systems?)
    - ## Cevap: Hayır (No)

### Önemli Noktalar
- ## RAG, üretken modellerin doğruluğunu artırmak için tasarlanmıştır.
- ## Naif RAG basit embedding teknikleri kullanır.
- ## RAG, gerçek zamanlı veri almak için harici kaynaklara erişebilir.
- ## Gelişmiş RAG konfigürasyonları çok modlu verileri işleyebilir.
- ## İnsan geri bildirimi RAG sistemlerinin değerlendirilmesinde önemlidir.

### Kod Örneği ve Açıklaması
RAG'ın nasıl çalıştığını göstermek için basit bir kod örneği:
```python
import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

## RAG modeli ve tokenizer yükleme
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

## Giriş metnini işleme
input_dict = tokenizer.prepare_seq2seq_batch("What is the capital of France?", return_tensors="pt")

## Cevap oluşturma
generated_ids = model.generate(input_ids=input_dict["input_ids"], attention_mask=input_dict["attention_mask"])

## Oluşturulan cevabı decode etme
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_text)
```
Bu kod, RAG modelini kullanarak bir soruya cevap üretir. 
- ## `RagTokenizer` ve `RagRetriever`, sırasıyla metni işlemek ve ilgili bilgileri çekmek için kullanılır.
- ## `RagSequenceForGeneration`, cevabı oluşturmak için kullanılan modeldir.
- ## `prepare_seq2seq_batch`, giriş metnini modele uygun hale getirir.
- ## `generate`, modelin cevabı oluşturmasını sağlar.
- ## `batch_decode`, oluşturulan cevabı insan tarafından okunabilir hale getirir.

### Kodun Kullanımı
1. ## Gerekli kütüphaneleri (`transformers`, `torch`) yükleyin.
2. ## RAG modelini ve tokenizer'ı yüklemek için `from_pretrained` metodunu kullanın.
3. ## Giriş metnini `prepare_seq2seq_batch` ile işleyin.
4. ## `generate` metoduyla cevabı oluşturun.
5. ## Oluşturulan cevabı `batch_decode` ile decode edin.

Bu örnek, RAG'ın temel kullanımını gösterir ve daha karmaşık uygulamalar için genişletilebilir.

---

## References

## Referanslar
Paragrafta anlatılan konu, bilgi yoğun NLP (Natural Language Processing - Doğal Dil İşleme) görevleri için kullanılan "Retrieval-Augmented Generation" (Alınan Bilgiyle Geliştirilmiş Üretim) yöntemi hakkındadır.

## Konu Hakkında
"Retrieval-Augmented Generation" yöntemi, büyük dil modellerinin (Large Language Models) performansını artırmak için kullanılan bir tekniktir. Bu yöntem, modelin ürettiği metinleri, harici bir bilgi kaynağından alınan bilgilerle zenginleştirmeyi amaçlar.

## Önemli Noktalar
* "Retrieval-Augmented Generation" yöntemi, bilgi yoğun NLP görevleri için kullanılır.
* Büyük dil modellerinin (Large Language Models) performansını artırmak için kullanılır.
* Modelin ürettiği metinleri, harici bir bilgi kaynağından alınan bilgilerle zenginleştirir.

## İlgili Çalışmalar
* "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" by Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al.: 
 https://arxiv.org/abs/2005.11401
* "Retrieval-Augmented Generation for Large Language Models: A Survey" by Yunfan Gao, Yun Xiong, Xinyu Gao, et al.: 
 https://arxiv.org/abs/2312.10997

## Kod Örnekleri
İlgili çalışmalarda kullanılan kod örnekleri verilmemektedir. Ancak, "Retrieval-Augmented Generation" yöntemini uygulamak için kullanılan bazı Python kütüphaneleri ve kodları aşağıda verilmiştir:
```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Model ve tokenizer yükleme
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Metin üretme
input_text = "Bu bir örnek metindir."
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids)

# Üretilen metni yazdırma
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
## Kod Açıklaması
Yukarıdaki kod örneğinde, önceden eğitilmiş bir "T5" modelini (AutoModelForSeq2SeqLM) kullanarak metin üretme işlemi gerçekleştirilmektedir. 
* `import torch` : PyTorch kütüphanesini içe aktarır.
* `from transformers import AutoModelForSeq2SeqLM, AutoTokenizer` : Transformers kütüphanesinden AutoModelForSeq2SeqLM ve AutoTokenizer sınıflarını içe aktarır.
* `model_name = "t5-base"` : Kullanılacak modelin adını belirler.
* `tokenizer = AutoTokenizer.from_pretrained(model_name)` : Belirtilen model için tokenizer'ı yükler.
* `model = AutoModelForSeq2SeqLM.from_pretrained(model_name)` : Belirtilen model için AutoModelForSeq2SeqLM modelini yükler.
* `input_text = "Bu bir örnek metindir."` : Üretilecek metnin girdisini belirler.
* `input_ids = tokenizer.encode(input_text, return_tensors="pt")` : Girdi metnini, modelin kabul ettiği formatta kodlar.
* `output = model.generate(input_ids)` : Modeli kullanarak metin üretir.
* `print(tokenizer.decode(output[0], skip_special_tokens=True))` : Üretilen metni, okunabilir formatta yazdırır.

## OpenAI Modelleri
OpenAI modelleri hakkında daha fazla bilgi için: https://platform.openai.com/docs/models

## Teknik Terimler
* NLP (Natural Language Processing - Doğal Dil İşleme)
* "Retrieval-Augmented Generation" (Alınan Bilgiyle Geliştirilmiş Üretim)
* Large Language Models (Büyük Dil Modelleri)

---

## Further reading

## Daha Fazla Okuma (Further Reading)
RAG (Retrieval-Augmented Generation) ile çalışan Generative AI (Üretken Yapay Zeka) şeffaflığının neden önerildiğini anlamak için aşağıdaki kaynaklara bakabilirsiniz.

## Konu Hakkında Genel Bilgi
RAG-driven Generative AI şeffaflığı, yapay zeka modellerinin karar verme süreçlerini daha anlaşılır ve güvenilir hale getirmeyi amaçlar. Bu, özellikle büyük dil modelleri (Large Language Models, LLM) için önemlidir.

## Önemli Noktalar
*   RAG-driven Generative AI şeffaflığı, modelin iç işleyişini daha anlaşılır hale getirir (Model Interpretability).
*   Şeffaflık, güvenilirliği artırır ve modelin hatalarını düzeltmeyi kolaylaştırır (Model Debugging).
*   Stanford Üniversitesi'nin "Foundation Model Transparency Index" çalışması, bu konudaki önemli bir kaynak olarak öne çıkmaktadır.

## Kaynaklar
*   https://hai.stanford.edu/news/introducing-foundation-model-transparency-index

## Kod Örneği ve Açıklaması
Bu kaynakta kod örneği bulunmamaktadır. Ancak RAG-driven Generative AI uygulamalarında kullanılan bazı temel kütüphaneler ve kod yapıları aşağıda verilmiştir.

### Python Kodu
```python
import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Model ve tokenizer yükleme
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

# Input hazırlama
input_dict = tokenizer.prepare_seq2seq_batch("What is the capital of France?", return_tensors="pt")

# Çıktı üretme
generated_ids = model.generate(input_ids=input_dict["input_ids"], attention_mask=input_dict["attention_mask"])

# Çıktıyı metne çevirme
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_text)
```

## Kod Açıklaması
*   `import torch`: PyTorch kütüphanesini içe aktarır. Bu, derin öğrenme modellerinin geliştirilmesi ve eğitilmesi için kullanılır (Deep Learning Framework).
*   `from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration`: Hugging Face Transformers kütüphanesinden RAG modeli için gerekli olan tokenizer, retriever ve model sınıflarını içe aktarır. Tokenizer, metni modele uygun forma çevirir (Text Tokenization). Retriever, ilgili bilgileri veritabanından çeker (Information Retrieval). Model, RAG tabanlı metin üretimini gerçekleştirir (Text Generation).
*   `tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")`: Önceden eğitilmiş RAG tokenizer modelini yükler.
*   `retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)`: Önceden eğitilmiş RAG retriever modelini yükler. `use_dummy_dataset=True` parametresi, gerçek veritabanı yerine dummy (sahte) veri kümesi kullanır.
*   `model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)`: Önceden eğitilmiş RAG sequence generation modelini retriever ile birlikte yükler.
*   `input_dict = tokenizer.prepare_seq2seq_batch("What is the capital of France?", return_tensors="pt")`: Girdi metnini ("What is the capital of France?") modele uygun forma çevirir ve PyTorch tensorları olarak döndürür.
*   `generated_ids = model.generate(input_ids=input_dict["input_ids"], attention_mask=input_dict["attention_mask"])`: Model, girdi metnine dayanarak çıktı IDs üretir (Text Generation).
*   `generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)`: Üretilen IDs'leri metne çevirir ve özel tokenları atlar.
*   `print(generated_text)`: Üretilen metni yazdırır.

## Notlar
*   Bu kod örneği, RAG-driven Generative AI modelinin temel kullanımını gösterir.
*   Gerçek dünya uygulamalarında, retriever için uygun bir veritabanı ve daha karmaşık girdi-çıktı işlemleri gerekebilir.

---

## Join our community on Discord

## Discord Topluluğumuza Katılın (Join our community on Discord)
Packt linki üzerinden Discord platformunda yazar ve diğer okuyucularla tartışmalara katılmak için topluluğumuza katılın. 

## Önemli Noktalar
- Yazar ve diğer okuyucularla tartışmalara katılmak
- Discord platformunda etkileşimde bulunmak
- Packt linki üzerinden topluluğa katılmak

## Discord Linki
https://www.packt.link/rag

## Kod Örneği Yok
Bu metinde kod örneği bulunmamaktadır.

## Açıklama
Paragraf, okuyucuları Packt linki üzerinden Discord platformunda yazar ve diğer okuyucularla tartışmalara katılmaya davet etmektedir. Discord linki https://www.packt.link/rag adresidir. 

## Teknik Terimler
- Discord: Topluluklarla etkileşimde bulunmak için kullanılan bir platform (Discord Platform)
- Packt: Teknik kitap ve kaynaklar yayınlayan bir şirket (Packt Publisher)

## Markdown Kullanımı
Yazı markdown formatında ##yazı şeklinde yazılmıştır.

## İlgili Bilgiler
Packt, teknik konularda kitaplar ve kaynaklar yayınlayan bir şirkettir. Discord ise topluluklarla etkileşimde bulunmak için kullanılan bir platformdur. Bu platformda yazar ve okuyucular tartışmalara katılabilir ve etkileşimde bulunabilirler.

---

