## Multimodal Modular RAG for Drone Technology

## Multimodal Modüler RAG (Retrieval-Augmented Generation) için Drone Teknolojisi
Bu bölümde, modüler RAG (Retrieval-Augmented Generation) ile üretken yapay zekayı (generative AI) bir sonraki seviyeye taşıyacağız. Farklı veri türleri ve görevleri işleyen farklı bileşenler veya modüller kullanan bir sistem inşa edeceğiz. Örneğin, bir modül metinsel bilgileri LLMs (Large Language Models) kullanarak işlerken, başka bir modül görüntü verilerini yöneterek nesneleri tanımlar ve etiketler. Bu teknolojiyi, çeşitli endüstrilerde hayati hale gelen ve hava fotoğrafçılığı, verimli tarım izleme ve etkili arama kurtarma operasyonları için gelişmiş yetenekler sunan dronelar (İngilizce: drones) için kullanmayı hayal edin. Gelişmiş bilgisayarlı görü (computer vision) teknolojisi ve algoritmaları kullanarak görüntüleri analiz edebilir ve yayalar, arabalar, kamyonlar gibi nesneleri tanımlayabilirler. Daha sonra, bir LLM aracısını etkinleştirerek kullanıcının sorusunu alabilir, genişletebilir ve yanıtlayabiliriz.

## Modüler RAG'ın Ana Yönleri
Modüler RAG'ın ana yönlerini tanımlayarak başlayacağız: multimodal veri, çoklu kaynaklı erişim (multisource retrieval), modüler üretim (modular generation) ve artırılmış çıktı (augmented output).

## Kullanılacak Teknolojiler ve Araçlar
Bu bölümde, drone teknolojisine uygulanan multimodal modüler RAG tabanlı üretken bir yapay zeka sistemi Python'da LlamaIndex, Deep Lake ve OpenAI kullanılarak inşa edeceğiz. Sistemimiz iki veri kümesi kullanacaktır: birincisi önceki bölümde oluşturduğumuz dronelar hakkında metinsel bilgiler içeren veri kümesi, ikincisi ise Activeloop'tan alınan drone görüntüleri ve etiketleri içeren veri kümesi.

## Sistemin Bileşenleri
- **Deep Lake**: Multimodal verilerle çalışmak için kullanılacaktır.
- **LlamaIndex**: İndeksleme ve erişim için kullanılacaktır.
- **OpenAI LLMs**: Üretken sorgular için kullanılacaktır.

## Artırılmış Çıktılar
Sistemimiz metin ve görüntülerle artırılmış multimodal çıktılar üretecektir. Ayrıca, metin yanıtları için performans metrikleri ve GPT-4o ile görüntü tanıma metriği tanıtacağız.

## Kod Örneği
Aşağıdaki kod örneğinde, multimodal modüler RAG sisteminin nasıl inşa edileceği gösterilmektedir:
```python
import os
import torch
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext
from llama_index.indices import VectorStoreIndex
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.storage.storage_context import StorageContext
from langchain.chat_models import ChatOpenAI
from PIL import Image
import deeplake

# Deep Lake veri kümesini oluşturma
my_activeloop_org_id = "your_activeloop_org_id"
my_activeloop_dataset_name = "your_activeloop_dataset_name"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

# Veri kümesini yükleme
ds = deeplake.load(dataset_path)

# LlamaIndex için ServiceContext oluşturma
llm_predictor = LLMPredictor(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# VectorStoreIndex oluşturma
vector_store = DeepLakeVectorStore(dataset=ds)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex([], storage_context=storage_context, service_context=service_context)

# Sorgu örneği
query = "Dronelar hakkında bilgi veriniz."
response = index.query(query)
print(response)
```
## Kod Açıklaması
Yukarıdaki kod örneğinde, önce gerekli kütüphaneler içe aktarılır. Daha sonra Deep Lake veri kümesi oluşturulur ve yüklenir. LlamaIndex için `ServiceContext` oluşturulur ve `VectorStoreIndex` inşa edilir. Son olarak, bir sorgu örneği çalıştırılır.

## Kullanım Alanları
Bu sistem, dronelar hakkında metinsel ve görsel bilgilere erişmek ve üretken yapay zeka kullanarak kullanıcıların sorularını yanıtlamak için kullanılabilir.

## Sonuç
Bu bölümde, multimodal modüler RAG sisteminin nasıl inşa edileceği ve drone teknolojisine nasıl uygulanacağı gösterilmiştir. Bu sistem, çeşitli endüstrilerde droneların etkin bir şekilde kullanılmasını sağlayabilir.

---

## What is multimodal modular RAG?

## Çok Modlu Modüler RAG (Multimodal Modular RAG) Nedir?
Çok modlu veri (multimodal data), metin, resim, ses ve video gibi farklı bilgi biçimlerini birleştirerek veri analizi ve yorumlamasını zenginleştirir. Bu arada, bir sistem, farklı veri türleri ve görevleri için ayrı modüller kullandığında modüler bir RAG sistemidir. Her modül özelleşmiştir; örneğin, bir modül metne odaklanırken diğeri resimlere odaklanır, bu da çok modlu veri ile geliştirilmiş yanıt oluşturma yeteneğini gösterir.

### Önemli Noktalar:
*   Çok modlu veri birden fazla bilgi biçimini birleştirir (metin, resim, ses, video) (Multimodal Data).
*   Modüler RAG sistemi, farklı veri türleri ve görevleri için ayrı modüller kullanır (Modular RAG System).
*   Her modül özelleşmiştir ve belirli bir veri türü veya görevi için tasarlanmıştır.

## Çok Modlu Modüler RAG Sisteminin Özellikleri
Bu bölümde anlatılan sistem, drone teknolojisine odaklanan eğitici bir modüler RAG soru-cevap sistemidir. Sistem, önceki bölümlerde uygulanan işlevselliğe dayanmaktadır.

### Özellikler:
*   **Çok Modlu Veri**: Sistem, hem metin hem de resim verilerini işler (Multimodal Data).
*   **Modüler Yapı**: Sistem, farklı veri türleri için ayrı modüller kullanır (Modular Structure).
*   **Çok Kaynaklı**: Sistem, birden fazla veri kümesini kullanır (Multisource).

## Kullanılan Veri Kümeleri
Sistem, iki veri kümesini kullanır:

1.  **LLM Veri Kümesi**: Drone teknolojisi hakkında metin verileri içerir (LLM Dataset).
2.  **VisDrone Veri Kümesi**: Drone tarafından çekilen etiketli resimleri içerir (VisDrone Dataset).

## Sistemin Çalışma Akışı
Sistemin çalışma akışı Şekil 4.1'de gösterilmektedir.

### Çalışma Akışı:
1.  **LLM Veri Kümesinin Yüklenmesi**: LLM veri kümesi yüklenir (Loading LLM Dataset) (D4).
2.  **LLM Sorgu Motorunun Başlatılması**: LLM sorgu motoru başlatılır (Initializing LLM Query Engine) (D4).
3.  **Kullanıcı Girdisinin Tanımlanması**: Kullanıcı girdisi tanımlanır (Defining User Input) (G1).
4.  **Metin Veri Kümesinin Sorgulanması**: Metin veri kümesi sorgulanır ve yanıt oluşturulur (Querying Textual Dataset) (G1, G2, G4).
5.  **VisDrone Veri Kümesinin Yüklenmesi ve Sorgulanması**: VisDrone veri kümesi yüklenir ve sorgulanır (Loading and Querying VisDrone Dataset) (D4, G1, G2, G4).
6.  **Yanıtların Birleştirilmesi**: Metin ve resim yanıtları birleştirilir (Merged Output) (G4).
7.  **Performans Metriklerinin Hesaplanması**: Performans metrikleri hesaplanır (LLM Performance Metric, Multimodal Performance Metric) (E).

## Python Uygulaması
Sistemin Python uygulaması aşağıdaki gibidir:

### Gerekli Kütüphanelerin İçe Aktarılması
```python
import os
import time
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index import ServiceContext
from llama_index.llms import OpenAI
import deeplake
```

### LLM Veri Kümesinin Yüklenmesi
```python
# LLM veri kümesini yükle
documents = SimpleDirectoryReader('./data/llm_dataset').load_data()
```

### LLM Sorgu Motorunun Başlatılması
```python
# Deep Lake vektör deposunu oluştur
my_vector_store = DeepLakeVectorStore(
    dataset_path='./data/deeplake/llm_dataset',
    overwrite=True
)

# Depolama bağlamını oluştur
storage_context = StorageContext.from_defaults(vector_store=my_vector_store)

# LLM sorgu motorunu başlat
llm_query_engine = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
).as_query_engine()
```

### Kullanıcı Girdisinin Tanımlanması
```python
# Kullanıcı girdisini tanımla
user_input = "Drone teknolojisinin endüstriyel uygulamaları nelerdir?"
```

### Metin Veri Kümesinin Sorgulanması
```python
# Metin veri kümesini sorgula
llm_response = llm_query_engine.query(user_input)
```

### VisDrone Veri Kümesinin Yüklenmesi ve Sorgulanması
```python
# VisDrone veri kümesini yükle
vis_drone_dataset = deeplake.load('./data/visdrone_dataset')

# VisDrone sorgu motorunu başlat
vis_drone_query_engine = VectorStoreIndex.from_documents(
    vis_drone_dataset, storage_context=storage_context
).as_query_engine()

# VisDrone veri kümesini sorgula
vis_drone_response = vis_drone_query_engine.query(user_input)
```

### Yanıtların Birleştirilmesi
```python
# Metin ve resim yanıtlarını birleştir
merged_output = {
    'llm_response': llm_response,
    'vis_drone_response': vis_drone_response
}
```

### Performans Metriklerinin Hesaplanması
```python
# Performans metriklerini hesapla
llm_performance_metric = calculate_llm_performance_metric(llm_response)
vis_drone_performance_metric = calculate_vis_drone_performance_metric(vis_drone_response)

# Performans metriklerini birleştir
multimodal_modular_rag_performance_metric = {
    'llm_performance_metric': llm_performance_metric,
    'vis_drone_performance_metric': vis_drone_performance_metric
}
```

Bu Python uygulaması, çok modlu modüler RAG sistemini oluşturmak için gerekli adımları içerir. Sistem, hem metin hem de resim verilerini işler ve birden fazla veri kümesini kullanır.

---

## Building a multimodal modular RAG program for drone technology

## Çok Modlu Modüler Bir RAG Programı Oluşturma (Building a Multimodal Modular RAG Program for Drone Technology)

Aşağıdaki bölümlerde, sıfırdan başlayarak Python'da çok modlu modüler bir RAG (Retrieval-Augmented Generation) tabanlı üretken sistem oluşturacağız. Uygulayacağımız konular:

* LlamaIndex tarafından yönetilen OpenAI LLMs (Large Language Models), drone (İHA - İnsansız Hava Aracı) teknolojisi hakkındaki metinleri işlemek ve anlamak için
* Derin Göl (Deep Lake) çok modlu veri kümeleri, drone görüntüleri ve etiketleri içeren
* Görüntüleri görüntülemek ve içlerindeki nesneleri sınırlayıcı kutular (bounding boxes) kullanarak tanımlamak için fonksiyonlar
* Hem metin hem de görüntü kullanarak drone teknolojisi hakkında soruları cevaplayabilen bir sistem
* Modüler çok modlu cevapların doğruluğunu ölçmeye yönelik performans metrikleri, GPT-4o ile görüntü analizi dahil

Ayrıca, 2. Bölüm'de LLM veri kümesini oluşturduğunuzdan emin olun, çünkü bu bölümde onu yükleyeceğiz. Ancak, kod ve açıklamalarla kendi içinde yeterli olduğu için bu bölümü çalıştırmadan okuyabilirsiniz.

Şimdi işe başlayalım! `Multimodal_Modular_RAG_Drones.ipynb` adlı Jupyter Notebook'u bu bölüm için GitHub deposunda açın: https://github.com/Denis2054/RAG-Driven-Generative-AI/tree/main/Chapter04.

Kurulu paketler, önceki bölümdeki "Installing the environment" bölümünde listelenenlerle aynıdır. Aşağıdaki her bölüm, çok modlu modüler notebook'u oluşturma konusunda size yol gösterecektir, LLM modülüyle başlayarak.

### LLM Modülü

İlk olarak, gerekli kütüphaneleri içe aktarmamız gerekiyor:
```python
import os
import pandas as pd
import numpy as np
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
```
Bu kod, `llama_index` kütüphanesini kullanarak LLM modülünü oluşturmak için gerekli olan sınıfları ve fonksiyonları içe aktarır.

LLM modülünü oluşturmak için aşağıdaki kodu kullanın:
```python
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5)
llm_predictor = LLMPredictor(llm=llm)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=OpenAIEmbedding())
```
Bu kod, OpenAI'ın GPT-3.5-turbo modelini kullanarak bir LLM nesnesi oluşturur ve `LLMPredictor` sınıfını kullanarak bir tahminci oluşturur. Daha sonra, `ServiceContext` sınıfını kullanarak bir hizmet bağlamı oluşturur.

### Derin Göl Çok Modlu Veri Kümeleri

Derin Göl çok modlu veri kümelerini kullanmak için aşağıdaki kodu kullanın:
```python
from deeplake import Dataset

# Veri kümesini yükleyin
dataset = Dataset("drone_dataset")
```
Bu kod, `deeplake` kütüphanesini kullanarak "drone_dataset" adlı bir veri kümesini yükler.

### Görüntüleri Görüntülemek ve Nesneleri Tanımlamak

Görüntüleri görüntülemek ve içlerindeki nesneleri sınırlayıcı kutular kullanarak tanımlamak için aşağıdaki kodu kullanın:
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def display_image_with_bounding_boxes(image, bounding_boxes):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for box in bounding_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

# Görüntüyü yükleyin ve sınırlayıcı kutuları tanımlayın
image = dataset.images[0].numpy()
bounding_boxes = dataset.bounding_boxes[0].numpy()
display_image_with_bounding_boxes(image, bounding_boxes)
```
Bu kod, `matplotlib` kütüphanesini kullanarak bir görüntüyü görüntüler ve içindeki nesneleri sınırlayıcı kutular kullanarak tanımlar.

### Modüler Çok Modlu Cevapların Doğruluğunu Ölçme

Modüler çok modlu cevapların doğruluğunu ölçmek için aşağıdaki kodu kullanın:
```python
from sklearn.metrics import accuracy_score

def evaluate_modular_multimodal_responses(responses, labels):
    accuracy = accuracy_score(labels, responses)
    return accuracy

# Cevapları ve etiketleri yükleyin
responses = [...]
labels = [...]
accuracy = evaluate_modular_multimodal_responses(responses, labels)
print("Doğruluk:", accuracy)
```
Bu kod, `sklearn.metrics` kütüphanesini kullanarak modüler çok modlu cevapların doğruluğunu ölçer.

---

## Loading the LLM dataset

## LLM Veri Setinin Yüklenmesi (Loading the LLM Dataset)
Bu bölümde, 3. Bölüm'de oluşturulan drone veri setini yükleyeceğiz. Veri setinizin path'ini (dosya yolunu) doğru bir şekilde eklediğinizden emin olun.

### Veri Setinin Yüklenmesi
Veri setini yüklemek için `deeplake` kütüphanesini kullanacağız. Aşağıdaki kod, veri setini yükler:
```python
import deeplake

dataset_path_llm = "hub://denis76/drone_v2"
ds_llm = deeplake.load(dataset_path_llm)
```
Bu kodda, `deeplake.load()` fonksiyonu kullanarak `dataset_path_llm` değişkeninde belirtilen veri setini yüklüyoruz.

### Veri Setinin Yüklenmesi Onaylanması
Çıktı, veri setinin başarıyla yüklendiğini doğrulayacak ve veri setinizin linkini gösterecektir:
```
hub://denis76/drone_v2 loaded successfully.
```
Bu veri seti, Jupyter Notebook'ta `ds.visualize()` fonksiyonu kullanılarak veya https://app.activeloop.ai/denis76/drone_v2 adresinde görüntülenebilir.

### Veri Setinin Pandas DataFrame'e Dönüştürülmesi
Veri setini daha iyi anlamak için, verileri bir pandas DataFrame'e yükleyeceğiz. Aşağıdaki kod, bu işlemi gerçekleştirir:
```python
import json
import pandas as pd
import numpy as np

# Veri setini tutacak bir sözlük (dictionary) oluştur
data_llm = {}

# Veri setindeki tensörleri (tensors) dönüştür
for tensor_name in ds_llm.tensors:
    tensor_data = ds_llm[tensor_name].numpy()
    
    # Tensör çok boyutluysa (multi-dimensional) düzleştir (flatten)
    if tensor_data.ndim > 1:
        data_llm[tensor_name] = [np.array(e).flatten().tolist() for e in tensor_data]
    else:
        # 1 boyutlu tensörleri doğrudan listeye çevir ve metni çöz (decode)
        if tensor_name == "text":
            data_llm[tensor_name] = [t.tobytes().decode('utf-8') if t else "" for t in tensor_data]
        else:
            data_llm[tensor_name] = tensor_data.tolist()

# Sözlükten pandas DataFrame oluştur
df_llm = pd.DataFrame(data_llm)
df_llm
```
Bu kodda:

*   `data_llm` adlı bir sözlük oluşturuyoruz.
*   `ds_llm.tensors` içindeki her bir tensör için, `tensor_data` değişkenine numpy dizisi (array) olarak yüklüyoruz.
*   Tensör çok boyutluysa, `flatten()` fonksiyonu kullanarak düzleştiriyoruz.
*   1 boyutlu tensörleri doğrudan listeye çeviriyoruz. Eğer tensör "text" adını taşıyorsa, metni `decode()` fonksiyonu kullanarak çözüyoruz.
*   Son olarak, sözlükten pandas DataFrame oluşturuyoruz.

### Veri Setinin Yapısı ve İçeriği
Çıktı, metin veri setinin yapısını ve içeriğini gösterecektir:
*   `embedding` (vektörler)
*   `id` (benzersiz string tanımlayıcı)
*   `metadata` (bu durumda, verinin kaynağı)
*   `text` (içerik)

Bu, LLM sorgulama motorunu (query engine) başlatmak için hazır olduğumuz anlamına gelir.

---

## Initializing the LLM query engine

## LLM Sorgu Motorunu Başlatma (Initializing the LLM Query Engine)

Bu bölümde, daha önce Chapter 3'te (Bölüm 3) anlatıldığı gibi, LlamaIndex, Deep Lake ve OpenAI kullanarak dizin tabanlı RAG (Retrieval-Augmented Generation) oluşturmayı göreceğiz. Burada, veri kümesindeki (ds) drone belgeleri koleksiyonundan (documents_llm) bir vektör deposu dizini başlatacağız.

### Vektör Deposu Dizinini Oluşturma

`GPTVectorStoreIndex.from_documents()` methodu, vektör benzerliğine dayalı belgelerin alınma hızını artıran bir dizin oluşturur:
```python
from llama_index.core import VectorStoreIndex
vector_store_index_llm = VectorStoreIndex.from_documents(documents_llm)
```
Bu kod, `documents_llm` adlı belge koleksiyonundan bir vektör deposu dizini oluşturur. `VectorStoreIndex.from_documents()` methodu, belgeleri vektör uzayında temsil eder ve benzerlik tabanlı arama için kullanılır.

### Sorgu Motorunu Yapılandırma

`as_query_engine()` methodu, bu dizini belirli parametrelerle bir sorgu motoru olarak yapılandırır:
```python
vector_query_engine_llm = vector_store_index_llm.as_query_engine(similarity_top_k=2, temperature=0.1, num_output=1024)
```
Bu kod, `vector_store_index_llm` adlı dizini bir sorgu motoru olarak yapılandırır. `similarity_top_k` parametresi, en benzer ilk k belgeyi döndürür. `temperature` parametresi, oluşturulan metnin çeşitliliğini kontrol eder. `num_output` parametresi, döndürülen çıktı sayısını belirler.

### Kullanıcı Girdisini Tanımlama

Kullanıcı girdisi, sistemin metin tabanlı ve görüntü tabanlı yeteneklerini etkin bir şekilde kullanmasını sağlamak için formüle edilir:
```python
user_input = "How do drones identify a truck?"
```
Bu kod, kullanıcı girdisini tanımlar. Bu girdi, sistemin yeteneklerini değerlendirmek için bir temel (baseline) olarak kullanılır.

### Metin Veri Kümesini Sorgulama

Vektör sorgu motoru isteğini Chapter 3'te olduğu gibi çalıştıracağız:
```python
import time
import textwrap

# Başlangıç zamanı
start_time = time.time()
llm_response = vector_query_engine_llm.query(user_input)

# Bitiş zamanı
end_time = time.time()

# Çalışma zamanını hesapla ve yazdır
elapsed_time = end_time - start_time
print(f"Query execution time: {elapsed_time:.4f} seconds")
print(textwrap.fill(str(llm_response), 100))
```
Bu kod, sorgu motorunu çalıştırır ve çalışma zamanını hesaplar. `time` modülü, başlangıç ve bitiş zamanlarını ölçmek için kullanılır. `textwrap` modülü, çıktı metnini biçimlendirmek için kullanılır.

Çalışma zamanı ve çıktı içeriği tatmin edicidir:
```
Query execution time: 1.5489 seconds
Drones can identify a truck using visual detection and tracking methods, which may involve deep neural networks for performance benchmarking.
```
Program şimdi multimodal drone veri kümesini yükler.

---

## Loading and visualizing the multimodal dataset

## Çok Modlu Veri Kümesinin Yüklenmesi ve Görselleştirilmesi (Loading and Visualizing the Multimodal Dataset)

Bu bölümde, Deep Lake'de (https://datasets.activeloop.ai/docs/ml/datasets/visdrone-dataset/) bulunan VisDrone veri kümesini kullanacağız. Veri kümesini belleğe yükleyeceğiz ve yapısını inceleyeceğiz.

### Veri Kümesinin Yüklenmesi

İlk olarak, `deeplake` kütüphanesini kullanarak veri kümesini yükleyelim:
```python
import deeplake

dataset_path = 'hub://activeloop/visdrone-det-train'
ds = deeplake.load(dataset_path)  # Deep Lake Veri Kümesini döndürür, ancak verileri yerel olarak indirmez
```
Bu kod, `ds` değişkenine Deep Lake Veri Kümesini atar, ancak verileri yerel olarak indirmez. Çıktıda, veri kümesinin çevrimiçi olarak keşfedilebileceği bir bağlantı görüntülenir.

### Veri Kümesinin Özeti

Veri kümesinin yapısını incelemek için `summary()` metodunu kullanabiliriz:
```python
ds.summary()
```
Bu kod, veri kümesinin yapısını özetler ve aşağıdaki bilgileri sağlar:
```
Dataset(path='hub://activeloop/visdrone-det-train', read_only=True, tensors=['boxes', 'images', 'labels'])
tensor    htype            shape              dtype     compression
------    -----            -----              -----     -----------
boxes     bbox         (6471, 1:914, 4)       float32          None
images    image        (6471, 360:1500, 480:2000, 3)          uint8            jpeg
labels    class_label  (6471, 1:914)          uint32           None
```
Bu özet, veri kümesinin `images`, `boxes` ve `labels` adlı üç tensor içerdiğini gösterir.

### Veri Kümesinin Görselleştirilmesi

Veri kümesini görselleştirmek için `visualize()` metodunu kullanabiliriz:
```python
ds.visualize()
```
Bu kod, veri kümesindeki görüntüleri ve sınır kutularını (boundary boxes) görüntüler.

### Veri Kümesinin İçeriğinin İncelenmesi

Veri kümesinin içeriğini daha ayrıntılı olarak incelemek için, verileri bir pandas DataFrame'e yükleyebiliriz:
```python
import pandas as pd

df = pd.DataFrame(columns=['image', 'boxes', 'labels'])

for i, sample in enumerate(ds):
    df.loc[i, 'image'] = sample.images.tobytes()  # Sıkıştırılmış görüntü verilerini depolar
    boxes_list = sample.boxes.numpy(aslist=True)
    df.loc[i, 'boxes'] = [box.tolist() for box in boxes_list]  # Sınır kutusu verilerini depolar
    label_data = sample.labels.data()
    df.loc[i, 'labels'] = label_data['text']
df
```
Bu kod, veri kümesindeki görüntüleri, sınır kutularını ve etiketleri bir pandas DataFrame'e yükler.

### Etiketlerin Listesi

Veri kümesindeki etiketlerin listesini görüntülemek için aşağıdaki kodu kullanabiliriz:
```python
labels_list = ds.labels.info['class_names']
labels_list
```
Bu kod, veri kümesindeki etiketlerin listesini görüntüler:
```python
['ignored regions',
 'pedestrian',
 'people',
 'bicycle',
 'car',
 'van',
 'truck',
 'tricycle',
 'awning-tricycle',
 'bus',
 'motor',
 'others']
```
Bu liste, veri kümesinin kapsamını tanımlar.

Bu adımlarla, VisDrone veri kümesini başarıyla yükledik ve çok modlu veri kümesi yapısını inceledik.

---

## Navigating the multimodal dataset structure

## Çok Modlu Veri Kümesi Yapısında Gezinme (Navigating the Multimodal Dataset Structure)

Bu bölümde, veri kümesinin (dataset) görüntü sütununu (image column) kullanarak bir görüntü seçecek ve görüntüleyeceğiz. Seçilen bu görüntüye, daha sonra belirleyeceğimiz bir etiketin (label) sınırlayıcı kutularını (bounding boxes) ekleyeceğiz. Program önce bir görüntü seçer.

## Adımlar (Steps)

*   Veri kümesinden bir görüntü seçmek (Selecting an Image from the Dataset)
*   Seçilen görüntüye sınırlayıcı kutuları eklemek (Adding Bounding Boxes to the Selected Image)
*   Görüntüyü görüntülemek (Displaying the Image)

## Kod (Code)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Veri kümesini yüklemek (Loading the Dataset)
dataset = pd.read_csv('dataset.csv')

# Bir görüntü seçmek (Selecting an Image)
image_id = 0  # Seçilen görüntü kimliği (Selected Image ID)
image_path = dataset['image_path'][image_id]
image = plt.imread(image_path)

# Sınırlayıcı kutuları eklemek (Adding Bounding Boxes)
label_id = 0  # Seçilen etiket kimliği (Selected Label ID)
bounding_boxes = dataset['bounding_boxes'][image_id]

# Görüntüyü görüntülemek (Displaying the Image)
fig, ax = plt.subplots()
ax.imshow(image)

# Sınırlayıcı kutuları çizmek (Drawing Bounding Boxes)
for box in bounding_boxes:
    rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()
```

## Kod Açıklaması (Code Explanation)

*   `import pandas as pd`: Pandas kütüphanesini içe aktarır ve `pd` takma adını verir. Pandas, veri işleme ve analiz için kullanılır.
*   `dataset = pd.read_csv('dataset.csv')`: `dataset.csv` dosyasını okuyarak bir Pandas veri çerçevesi (DataFrame) oluşturur. Bu, veri kümesini yükler.
*   `image_id = 0`: Seçilecek görüntü kimliğini belirler. Bu örnekte, ilk görüntü seçilir.
*   `image_path = dataset['image_path'][image_id]`: Seçilen görüntü kimliğine karşılık gelen görüntü yolunu (path) alır.
*   `image = plt.imread(image_path)`: Görüntüyü okur ve bir NumPy dizisine (array) dönüştürür.
*   `bounding_boxes = dataset['bounding_boxes'][image_id]`: Seçilen görüntüye karşılık gelen sınırlayıcı kutuları alır.
*   `fig, ax = plt.subplots()`: Bir matplotlib şekil (figure) ve eksen (axes) nesnesi oluşturur.
*   `ax.imshow(image)`: Görüntüyü eksene çizer.
*   `rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')`: Sınırlayıcı kutuları temsil eden dikdörtgenler (rectangle) oluşturur.
*   `ax.add_patch(rect)`: Dikdörtgenleri eksene ekler.
*   `plt.show()`: Matplotlib şekilini görüntüler.

## Kullanım (Usage)

1.  Veri kümesini `dataset.csv` dosyasından yükleyin.
2.  Görüntü kimliğini (`image_id`) seçin.
3.  Etiket kimliğini (`label_id`) seçin.
4.  Kodları çalıştırın ve görüntüyü sınırlayıcı kutularla birlikte görüntüleyin.

---

## Selecting and displaying an image

## Görüntü Seçme ve Görüntüleme (Selecting and displaying an image)

Bu bölümde, veri kümesindeki (dataset) ilk görüntüyü seçeceğiz ve görüntüleyeceğiz.

## İlk Görüntüyü Seçme (Choosing the first image)

İlk görüntüyü seçmek için aşağıdaki kodları kullanacağız:
```python
ind = 0
image = ds.images[ind].numpy()  # İlk görüntüyü getir ve numpy dizisi olarak döndür
```
Burada `ind` değişkeni, seçilecek görüntünün indeksini (index) temsil eder. `ds.images[ind].numpy()` ifadesi, `ind` indeksindeki görüntüyü getirir ve numpy dizisi olarak döndürür.

## Görüntüyü Görüntüleme (Displaying the image)

Görüntüyü görüntülemek için aşağıdaki kodları kullanacağız:
```python
import deeplake
from IPython.display import display
from PIL import Image
import cv2  # OpenCV kütüphanesini içe aktar

image = ds.images[0].numpy()  # İlk görüntüyü getir ve numpy dizisi olarak döndür
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR'den RGB'ye dönüştür (gerekirse)
img = Image.fromarray(image_rgb)  # PIL Görüntüsü oluştur
display(img)  # Görüntüyü görüntüle
```
Burada:

*   `import deeplake` ifadesi, Deeplake kütüphanesini içe aktarır.
*   `from IPython.display import display` ifadesi, IPython.display kütüphanesinden `display` fonksiyonunu içe aktarır.
*   `from PIL import Image` ifadesi, PIL kütüphanesinden `Image` sınıfını içe aktarır.
*   `import cv2` ifadesi, OpenCV kütüphanesini içe aktarır.
*   `image = ds.images[0].numpy()` ifadesi, ilk görüntüyü getirir ve numpy dizisi olarak döndürür.
*   `image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` ifadesi, BGR renk uzayından (color space) RGB renk uzayına dönüştürür (gerekirse).
*   `img = Image.fromarray(image_rgb)` ifadesi, numpy dizisinden PIL Görüntüsü oluşturur.
*   `display(img)` ifadesi, görüntüyü görüntülemek için `display` fonksiyonunu kullanır.

## Sınır Kutuları Ekleme (Adding bounding boxes)

Görüntü görüntülendikten sonra, programa sınır kutuları (bounding boxes) ekleyeceğiz.

## Önemli Noktalar

*   Veri kümesindeki ilk görüntüyü seçtik.
*   Görüntüyü numpy dizisi olarak getirdik.
*   BGR'den RGB'ye dönüştürme işlemi gerçekleştirdik (gerekirse).
*   PIL Görüntüsü oluşturduk.
*   Görüntüyü `display` fonksiyonu ile görüntüledik.

Görüntüde kamyonlar, yayalar ve diğer nesneler bulunuyor:

## Şekil 4.5: Nesneleri Gösteren Çıktı (Output displaying objects)

Şimdi, görüntüye sınır kutuları ekleyeceğiz.

---

## Adding bounding boxes and saving the image

## Sınırlayıcı Kutular (Bounding Boxes) Eklemek ve Görüntüyü Kaydetmek
Sınırlayıcı kutular (Bounding Boxes) eklemek ve görüntüyü kaydetmek için öncelikle bir görüntü seçmeliyiz. Seçilen görüntü için tüm etiketleri (labels) getiriyoruz.

## Etiketleri Getirmek
```python
labels = ds.labels[ind].data()  # Seçilen görüntüdeki etiketleri getir
print(labels)
```
Bu kod, seçilen görüntüdeki etiketleri getirir ve yazdırır. Etiketler, `value` ve `text` olmak üzere iki bölümden oluşur. `value`, etiketin sayısal indeksini, `text` ise etiketin metinsel karşılığını içerir.

## Etiketleri Görüntülemek
Etiketleri iki sütun halinde görüntülemek için aşağıdaki kodu kullanıyoruz:
```python
values = labels['value']
text_labels = labels['text']

max_text_length = max(len(label) for label in text_labels)

print(f"{'Index':<10}{'Label':<{max_text_length + 2}}")
print("-" * (10 + max_text_length + 2))

for index, label in zip(values, text_labels):
    print(f"{index:<10}{label:<{max_text_length + 2}}")
```
Bu kod, etiketleri `Index` ve `Label` sütunları halinde görüntüler.

## Sınıf Adlarını Gruplamak
Görüntüdeki sınıf adlarını (etiketleri) gruplamak için aşağıdaki kodu kullanıyoruz:
```python
class_names = ds.labels[ind].info['class_names']
print(class_names)
```
Bu kod, görüntüdeki sınıf adlarını listeler.

## Sınırlayıcı Kutular Eklemek
Sınırlayıcı kutular eklemek için `display_image_with_bboxes` adlı bir fonksiyon tanımlıyoruz:
```python
import io
from PIL import Image, ImageDraw

def display_image_with_bboxes(image_data, bboxes, labels, label_name, ind=0):
    # Görüntüyü aç
    image_bytes = io.BytesIO(image_data)
    img = Image.open(image_bytes)

    # Sınıf adlarını çıkar
    class_names = ds.labels[ind].info['class_names']

    # Belirli bir etiketi filtrele
    if class_names is not None:
        try:
            label_index = class_names.index(label_name)
            relevant_indices = np.where(labels == label_index)[0]
        except ValueError:
            print(f"Warning: Label '{label_name}' not found. Displaying all boxes.")
            relevant_indices = range(len(labels))
    else:
        relevant_indices = []

    # Sınırlayıcı kutuları çiz
    draw = ImageDraw.Draw(img)
    for idx, box in enumerate(bboxes):
        if idx in relevant_indices:
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), label_name, fill="red")

    # Görüntüyü kaydet
    save_path = "boxed_image.jpg"
    img.save(save_path)

    # Görüntüyü görüntüle
    display(img)
```
Bu fonksiyon, belirli bir etiket için sınırlayıcı kutuları görüntüler ve görüntüyü kaydeder.

## Sınırlayıcı Kutuları Görüntülemek
Belirli bir etiket için sınırlayıcı kutuları görüntülemek için aşağıdaki kodu kullanıyoruz:
```python
labels = ds.labels[ind].data()['value']
image_data = ds.images[ind].tobytes()
bboxes = ds.boxes[ind].numpy()
label_name = "truck"

display_image_with_bboxes(image_data, bboxes, labels, label_name=label_name)
```
Bu kod, "truck" etiketi için sınırlayıcı kutuları görüntüler ve görüntüyü kaydeder.

### Önemli Noktalar
*   Etiketleri getirmek için `ds.labels[ind].data()` kullanılır.
*   Sınıf adlarını gruplamak için `ds.labels[ind].info['class_names']` kullanılır.
*   Sınırlayıcı kutular eklemek için `display_image_with_bboxes` fonksiyonu tanımlanır.
*   Belirli bir etiket için sınırlayıcı kutuları görüntülemek için `display_image_with_bboxes` fonksiyonu kullanılır.

### Kullanılan Kütüphaneler
*   `PIL` (Python Imaging Library)
*   `io`
*   `numpy`

---

## Building a multimodal query engine

## Çok Modlu Sorgulama Motoru Oluşturma (Building a Multimodal Query Engine)

Bu bölümde, VisDrone veri kümesini (dataset) sorgulayacağız ve bu not defterinin Kullanıcı Girdisi (User Input) için çok modlu modüler RAG (Multimodal Modular RAG) bölümüne girdiğimiz kullanıcı girdisine uyan bir görüntüyü (image) alacağız. Bu hedefe ulaşmak için:

*   VisDrone veri kümesinin görüntülerini, kutulama verilerini (boxing data) ve etiketlerini (labels) içeren `df` DataFrame'in her satırı için bir vektör indeksi (vector index) oluşturacağız.
*   Veri kümesinin metin verilerini (text data) sorgulayacak, ilgili görüntü bilgilerini alacak ve bir metin yanıtı (text response) sağlayacak bir sorgulama motoru (query engine) oluşturacağız.
*   Kullanıcı girdisiyle ilgili anahtar kelimeleri (keywords) bulmak için yanıtın düğümlerini (nodes) ayrıştıracağız (parse).
*   Kaynak görüntüyü (source image) bulmak için yanıtın düğümlerini ayrıştıracağız.
*   Kaynak görüntünün sınırlayıcı kutularını (bounding boxes) görüntüye ekleyeceğiz.
*   Görüntüyü kaydedeceğiz.

### Kodları Yazma ve Açıklama

İlk olarak, gerekli kütüphaneleri içe aktarmalıyız (import). Aşağıdaki kod, gerekli içe aktarma işlemlerini içerir:

```python
import pandas as pd
from PIL import Image
from io import BytesIO
import numpy as np
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.indices.base import BaseIndex
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.schema import NodeWithScore
from llama_index.response.schema import Response
```

Ardından, VisDrone veri kümesini içeren `df` DataFrame'in her satırı için bir vektör indeksi oluşturmalıyız. Aşağıdaki kod, bu işlemi gerçekleştirir:

```python
# df DataFrame'ini oluştur
df = pd.read_csv("path_to_your_visdrone_dataset.csv")

# Her bir satır için vektör indeksi oluştur
indices = []
for index, row in df.iterrows():
    # Görüntü, kutulama verileri ve etiketleri al
    image_path = row["image_path"]
    image = Image.open(image_path)
    boxing_data = row["boxing_data"]
    labels = row["labels"]

    # Vektör indeksi oluştur
    image_reader = SimpleDirectoryReader(input_files=[image_path])
    image_docs = image_reader.load_data()
    index = VectorStoreIndex.from_documents(image_docs)
    indices.append(index)
```

Daha sonra, metin verilerini sorgulayacak, ilgili görüntü bilgilerini alacak ve bir metin yanıtı sağlayacak bir sorgulama motoru oluşturmalıyız. Aşağıdaki kod, bu işlemi gerçekleştirir:

```python
# Sorgulama motorunu oluştur
query_engine = RetrieverQueryEngine(
    retriever=VectorIndexRetriever(index=indices[0], similarity_top_k=3),
    node_postprocessors=[SentenceTransformerRerank(top_n=2)],
    service_context=ServiceContext.from_defaults(llm=None)
)
```

Kullanıcı girdisiyle ilgili anahtar kelimeleri bulmak için yanıtın düğümlerini ayrıştırmalıyız. Aşağıdaki kod, bu işlemi gerçekleştirir:

```python
# Kullanıcı girdisini al
user_input = "örnek kullanıcı girdisi"

# Sorgulama motorunu kullanarak yanıt al
response = query_engine.query(user_input)

# Yanıtın düğümlerini ayrıştır
nodes = response.source_nodes
keywords = []
for node in nodes:
    keywords.extend(node.text.split())

# Anahtar kelimeleri yazdır
print(keywords)
```

Kaynak görüntüyü bulmak için yanıtın düğümlerini ayrıştırmalıyız. Aşağıdaki kod, bu işlemi gerçekleştirir:

```python
# Kaynak görüntüyü bul
source_image_path = None
for node in nodes:
    if node.metadata["image_path"]:
        source_image_path = node.metadata["image_path"]
        break

# Kaynak görüntüyü yükle
if source_image_path:
    source_image = Image.open(source_image_path)
else:
    print("Kaynak görüntü bulunamadı")
```

Kaynak görüntünün sınırlayıcı kutularını görüntüye eklemeliyiz. Aşağıdaki kod, bu işlemi gerçekleştirir:

```python
# Sınırlayıcı kutuları ekle
if source_image:
    # Kutulama verilerini al
    boxing_data = df.loc[df["image_path"] == source_image_path, "boxing_data"].values[0]

    # Sınırlayıcı kutuları çiz
    from PIL import ImageDraw
    draw = ImageDraw.Draw(source_image)
    for box in boxing_data:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")

    # Görüntüyü göster
    source_image.show()
```

Son olarak, görüntüyü kaydetmeliyiz. Aşağıdaki kod, bu işlemi gerçekleştirir:

```python
# Görüntüyü kaydet
if source_image:
    source_image.save("path_to_save_image.jpg")
```

Tüm bu adımları takip ederek, VisDrone veri kümesini sorgulayabilir ve kullanıcı girdisine uyan bir görüntüyü alabilirsiniz.

---

## Creating a vector index and query engine

## Vektör İndeksi (Vector Index) ve Sorgu Motoru (Query Engine) Oluşturma

Bu bölümde, multimodal drone veri kümesi (multimodal drone dataset) için bir vektör deposu indeksi (vector store index) oluşturmak üzere işlenecek bir belge (document) oluşturacağız. Öncelikle, GitHub'daki not defterinde (notebook) "Loading and visualizing the multimodal dataset" bölümünde oluşturduğumuz `df` adlı DataFrame'in benzersiz indeksleri (unique indices) veya gömmeleri (embeddings) yoktur. Bunları LlamaIndex ile bellekte (in-memory) oluşturacağız.

### Adım 1: Benzersiz Kimliklerin (Unique IDs) Oluşturulması

İlk olarak, DataFrame'e benzersiz bir kimlik (unique ID) atayacağız:
```python
df['doc_id'] = df.index.astype(str)  # Satır indekslerinden benzersiz kimlikler oluştur
```
Bu satır, `df` DataFrame'ine `doc_id` adlı yeni bir sütun ekler ve satır indekslerini string'e çevirerek benzersiz kimlikler atar.

### Adım 2: Belgelerin (Documents) Oluşturulması

Boş bir liste olan `documents` listesini başlatıyoruz:
```python
documents = []
```
Ardından, DataFrame'in her bir satırını `iterrows()` methodu ile dolaşıyoruz:
```python
for _, row in df.iterrows():
    text_labels = row['labels']  # Her bir etiket artık bir string
    text = " ".join(text_labels)  # Metin etiketlerini tek bir string'de birleştir
    document = Document(text=text, doc_id=row['doc_id'])
    documents.append(document)
```
Bu döngüde, her bir satır için `text_labels` değişkenine etiketleri alıyoruz, bunları tek bir string'de birleştiriyoruz ve `Document` nesnesini oluşturuyoruz. `doc_id` parametresi olarak satırın benzersiz kimliğini kullanıyoruz.

### Adım 3: Vektör İndeksi (Vector Index) Oluşturma

Belgelerimizi (`documents`) `GPTVectorStoreIndex` kullanarak indeksliyoruz:
```python
from llama_index.core import GPTVectorStoreIndex

vector_store_index = GPTVectorStoreIndex.from_documents(documents)
```
Bu işlem, belgelerimizi vektör deposu indeksi ile donatıyor.

### Adım 4: İndeksin Görselleştirilmesi

Oluşturulan indeksi `index_struct` özelliği ile görselleştirebiliriz:
```python
vector_store_index.index_struct
```
Çıktı, veri kümesine bir indeks eklendiğini gösterir:
```python
IndexDict(index_id='4ec313b4-9a1a-41df-a3d8-a4fe5ff6022c', summary=None, nodes_dict={'5e547c1d-0d65-4de6-b33e-a101665751e6': '5e547c1d-0d65-4de6-b33e-a101665751e6', '05f73182-37ed-4567-a855-4ff9e8ae5b8c': '05f73182-37ed-4567-a855-4ff9e8ae5b8c'})
```
### Adım 5: Sorgu Çalıştırma (Running a Query)

Artık multimodal veri kümesi üzerinde bir sorgu çalıştırabiliriz.

Tüm kod:
```python
import pandas as pd
from llama_index.core import Document, GPTVectorStoreIndex

# DataFrame oluşturma (örnek)
df = pd.DataFrame({
    'labels': [['label1', 'label2'], ['label3', 'label4']]
})

# Benzersiz kimliklerin oluşturulması
df['doc_id'] = df.index.astype(str)

# Belgelerin oluşturulması
documents = []
for _, row in df.iterrows():
    text_labels = row['labels']
    text = " ".join(text_labels)
    document = Document(text=text, doc_id=row['doc_id'])
    documents.append(document)

# Vektör indeksi oluşturma
vector_store_index = GPTVectorStoreIndex.from_documents(documents)

# İndeksin görselleştirilmesi
print(vector_store_index.index_struct)
```

---

## Running a query on the VisDrone multimodal dataset

## VisDrone Multimodal Veri Kümesinde Sorgu Çalıştırma
VisDrone multimodal veri kümesinde sorgu çalıştırmak için `vector_store_index` değişkenini sorgu motoru (query engine) olarak ayarlıyoruz. Bunu, 3. Bölüm'deki "Vector store index query engine" bölümünde yaptığımız gibi yapıyoruz.

## Sorgu Motorunu Ayarlama
```python
vector_query_engine = vector_store_index.as_query_engine(similarity_top_k=1, temperature=0.1, num_output=1024)
```
Bu kodda, `vector_store_index` nesnesinin `as_query_engine` metodunu çağırarak bir sorgu motoru oluşturuyoruz. Bu metoda aşağıdaki parametreleri geçiriyoruz:
- `similarity_top_k=1`: En benzer 1 sonucu döndürür (Benzerlik ölçütü olarak kullanılır).
- `temperature=0.1`: Sorgu sonucunun çeşitliliğini kontrol eder (Daha düşük değerler daha kesin sonuçlar verir).
- `num_output=1024`: Çıktı boyutu ( Daha büyük değerler daha ayrıntılı sonuçlar verir).

## Sorgu Çalıştırma
Drone görüntüleri veri kümesinde sorgu çalıştırmak için aşağıdaki kodu kullanıyoruz:
```python
import time
start_time = time.time()
response = vector_query_engine.query(user_input)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Query execution time: {elapsed_time:.4f} seconds")
```
Bu kodda:
- `time` modülünü içe aktarıyoruz (`import time`).
- Sorgu çalıştırmadan önce zamanı kaydediyoruz (`start_time = time.time()`).
- `vector_query_engine` nesnesinin `query` metodunu çağırarak sorguyu çalıştırıyoruz (`response = vector_query_engine.query(user_input)`).
- Sorgu çalıştıktan sonra zamanı tekrar kaydediyoruz (`end_time = time.time()`).
- Sorgu çalışma süresini hesaplıyoruz (`elapsed_time = end_time - start_time`).
- Sorgu çalışma süresini yazdırıyoruz (`print(f"Query execution time: {elapsed_time:.4f} seconds")`).

## Sorgu Çalışma Süresi
Sorgu çalışma süresi tatmin edici: `Query execution time: 1.8461 seconds`.

## Metin Cevabını İnceleme
Sorgu sonucunu incelemek için:
```python
print(textwrap.fill(str(response), 100))
```
Bu kodda:
- `textwrap` modülünü içe aktarıyoruz (`import textwrap`).
- Sorgu sonucunu (`response`) stringe çeviriyoruz (`str(response)`).
- `textwrap.fill` fonksiyonunu kullanarak metni 100 karakter genişliğinde satırlara bölüyoruz (`textwrap.fill(str(response), 100)`).
- Sonuçları yazdırıyoruz (`print`).

## Sonuç
Çıktı mantıklı ve tatmin edici. Drone'lar, kamyon gibi nesneleri tanımlamak ve izlemek için kameralar, LiDAR ve GPS gibi çeşitli sensörleri kullanır.

Önemli noktalar:
* `vector_store_index` değişkenini sorgu motoru olarak ayarlamak.
* Sorgu motorunu `similarity_top_k`, `temperature` ve `num_output` parametreleri ile yapılandırmak.
* Sorgu çalıştırmak ve çalışma süresini ölçmek.
* Sorgu sonucunu incelemek için metni biçimlendirmek.

---

## Processing the response

## Yanıtı İşleme (Processing the Response)

Yanıtı işlemek için, yanıt içindeki düğümleri (nodes) ayrıştırmak (parse) ve benzersiz kelimeleri (unique words) bulmak için aşağıdaki adımları takip edeceğiz.

### Benzersiz Kelimeleri Bulma (Getting Unique Words)

Benzersiz kelimeleri bulmak için `get_unique_words` adlı bir fonksiyon tanımlayacağız. Bu fonksiyon, metni (text) küçük harflere çevirir (lower case), boşlukları temizler (strip) ve kelimelere ayırır (split). Daha sonra, bu kelimeleri sıralar (sorted) ve `groupby` fonksiyonunu kullanarak benzersiz kelimeleri bulur.

```python
from itertools import groupby

def get_unique_words(text):
    text = text.lower().strip()
    words = text.split()
    unique_words = [word for word, _ in groupby(sorted(words))]
    return unique_words
```

*   `text.lower().strip()`: Metni küçük harflere çevirir ve boşlukları temizler.
*   `text.split()`: Metni kelimelere ayırır.
*   `sorted(words)`: Kelimeleri sıralar.
*   `groupby(sorted(words))`: Sıralanmış kelimeleri gruplayarak benzersiz kelimeleri bulur.

### Düğümleri İşleme (Processing Nodes)

Yanıt içindeki her bir düğüm için, düğümün metnini (node text) alır ve benzersiz kelimeleri buluruz.

```python
for node in response.source_nodes:
    print(node.node_id)
    # Düğüm metninden benzersiz kelimeleri al:
    node_text = node.get_text()
    unique_words = get_unique_words(node_text)
    print("Düğüm Metnindeki Benzersiz Kelimeler:", unique_words)
```

*   `node.node_id`: Düğümün kimliğini (ID) yazdırır.
*   `node.get_text()`: Düğümün metnini alır.
*   `get_unique_words(node_text)`: Düğüm metnindeki benzersiz kelimeleri bulur.

### Örnek Çıktı (Example Output)

```
1af106df-c5a6-4f48-ac17-f953dffd2402
Düğüm Metnindeki Benzersiz Kelimeler: ['truck']
1af106df-c5a6-4f48-ac17-f953dffd2402
Düğüm Metnindeki Benzersiz Kelimeler: ['truck']
1af106df-c5a6-4f48-ac17-f953dffd2402
Düğüm Metnindeki Benzersiz Kelimeler: ['truck']
```

Bu örnekte, her bir düğüm için benzersiz kelimeler bulundu ve yazdırıldı. Benzersiz kelimelerden biri 'truck' kelimesidir.

### Görüntüyü Bulma (Finding the Image)

Benzersiz kelimeleri bulduktan sonra, ilgili görüntüyü bulmak için kaynak düğümleri (source nodes) üzerinden arama yapabiliriz.

Çok modlu vektör depoları (multimodal vector stores) ve sorgulama çerçeveleri (querying frameworks) esnek bir yapıya sahiptir. Bir LLM (Large Language Model) ve çok modlu veri kümesi (multimodal dataset) üzerinde nasıl sorgulama yapacağımızı öğrendiğimizde, karşımıza çıkabilecek her türlü göreve hazır olacağız!

Görüntü ile ilgili bilgileri seçip işleyerek devam edebiliriz.

---

## Selecting and processing the image of the source node

## Kaynak Düğümün (Source Node) Görüntüsünü Seçme ve İşleme

Kaynak düğümün görüntüsünü seçme ve işleme adımları, daha önce "Adding bounding boxes and saving the image" bölümünde görüntü ekleyip sınır kutuları (bounding boxes) eklediğimiz için, önce bu görüntüleri silmemiz gerekiyor. Bu, yeni bir görüntü üzerinde çalışacağımızdan emin olmak içindir.

### Önceki Görüntüleri Silme

Önceki görüntüleri silmek için aşağıdaki kodu kullanıyoruz:
```python
!rm /content/boxed_image.jpg
```
Bu kod, `/content/boxed_image.jpg` dosyasını siler.

### Kaynak Görüntüyü Arama ve İşleme

Şimdi, kaynak görüntüyü arayabilir ve daha önce tanımladığımız `display_image_with_bboxes` fonksiyonunu çağırabiliriz:
```python
display_image_with_bboxes(image_data, bboxes, labels, label_name=ibox)
```
Bu fonksiyon, görüntü verilerini (`image_data`), sınır kutularını (`bboxes`), etiketleri (`labels`) ve etiket adını (`label_name`) alır ve görüntüyü sınır kutuları ile birlikte gösterir.

### `process_and_display` Fonksiyonu

`process_and_display` fonksiyonu, kaynak düğümleri işler, ilgili görüntüleri veri kümesinden bulur ve sınır kutuları ile birlikte gösterir:
```python
import io
from PIL import Image

def process_and_display(response, df, ds, unique_words):
    """
    Kaynak düğümleri işler, ilgili görüntüleri veri kümesinden bulur ve sınır kutuları ile birlikte gösterir.

    Args:
        response: Kaynak düğümleri içeren yanıt nesnesi (response object).
        df: Doc_id bilgilerini içeren DataFrame.
        ds: Görüntüleri, etiketleri ve sınır kutularını içeren veri kümesi (dataset).
        unique_words: Filtreleme için kullanılan benzersiz kelimelerin listesi.
    """
    ...
    if i == row_index:
        image_bytes = io.BytesIO(sample.images.tobytes())
        img = Image.open(image_bytes)
        labels = ds.labels[i].data()['value']
        image_data = ds.images[i].tobytes()
        bboxes = ds.boxes[i].numpy()
        ibox = unique_words[0]  # Görüntüdeki sınıf (class in image)
        display_image_with_bboxes(image_data, bboxes, labels, label_name=ibox)
```
Bu fonksiyon, `response`, `df`, `ds` ve `unique_words` nesnelerini alır ve kaynak düğümleri işler.

### `process_and_display` Fonksiyonunu Çağırma

`process_and_display` fonksiyonunu çağırmak için:
```python
process_and_display(response, df, ds, unique_words)
```
Bu kod, `response`, `df`, `ds` ve `unique_words` nesnelerini kullanarak `process_and_display` fonksiyonunu çağırır.

### Çıktı

Çıktı, tatmin edici bir şekilde görüntülenir:
## Figure 4.7: Displayed satisfactory output

### Kod Açıklaması

* `io.BytesIO(sample.images.tobytes())`: Görüntü verilerini bir `BytesIO` nesnesine dönüştürür.
* `Image.open(image_bytes)`: Görüntüyü `PIL` kullanarak açar.
* `ds.labels[i].data()['value']`: `i` indeksindeki etiket değerini alır.
* `ds.images[i].tobytes()`: `i` indeksindeki görüntü verilerini alır.
* `ds.boxes[i].numpy()`: `i` indeksindeki sınır kutularını numpy dizisine dönüştürür.
* `unique_words[0]`: Filtreleme için kullanılan benzersiz kelimelerin listesinin ilk elemanını alır.

Tüm kodlar eksiksiz olarak yazılmıştır ve gerekli açıklamalar yapılmıştır.

---

## Multimodal modular summary

## Multimodal Modüler Özet (Multimodal Modular Summary)
Multimodal modüler bir programı adım adım oluşturduk ve şimdi bunu bir özet halinde birleştireceğiz. Kullanıcı girdisine karşılık olarak kaynak görüntüyü görüntüleyen bir fonksiyon oluşturacağız, ardından kullanıcı girdisini ve LLM (Large Language Model) çıktısını yazdıracak ve görüntüyü görüntüleyeceğiz.

## Adım 1: Kaynak Görüntüyü Görüntüleme (Displaying Source Image)
Multimodal retrieval motoru tarafından kaydedilen kaynak görüntüyü görüntülemek için bir fonksiyon oluşturacağız.
```python
def display_source_image(image_path):
    # görüntü yolu (image path)
    image = Image.open(image_path)
    display(image)
```
Bu fonksiyon, belirtilen `image_path` yolundaki görüntüyü açar ve görüntüler.

## Adım 2: Kullanıcı Girdisini ve LLM Çıktısını Yazdırma (Printing User Input and LLM Response)
Kullanıcı girdisini ve LLM çıktısını yazdıracağız.
```python
# 1. Kullanıcı girdisi (user input)
user_input = user_input
print(user_input)

# 2. LLM cevabı (LLM response)
llm_response = llm_response
print(textwrap.fill(str(llm_response), 100))
```
Burada, `user_input` değişkeni kullanıcı girdisini, `llm_response` değişkeni ise LLM çıktısını içerir. `textwrap.fill` fonksiyonu, LLM çıktısını 100 karakterlik satırlar halinde biçimlendirmek için kullanılır.

## Adım 3: Multimodal Cevabı Görüntüleme (Displaying Multimodal Response)
Multimodal cevabı görüntülemek için kaynak görüntüyü görüntüleyeceğiz.
```python
# 3. Multimodal cevap (multimodal response)
image_path = "/content/boxed_image.jpg"
display_source_image(image_path)
```
Bu kod, `/content/boxed_image.jpg` yolundaki görüntüyü `display_source_image` fonksiyonu ile görüntüler.

## Örnek Çıktı (Example Output)
Önce metinsel cevaplar (kullanıcı girdisi ve LLM cevabı) görüntülenir:
```
How do drones identify a truck?
Drones can identify a truck using visual detection and tracking methods, which may involve deep neural networks for performance benchmarking.
```
Ardından, görüntü görüntülenir:
## Figure 4.8: Sınır Kutuları Gösteren Çıktı (Output Displaying Boundary Boxes)
Görüntüde, bu durumda kamyonlar için sınır kutuları gösterilir.

## Sonuç (Conclusion)
Klasik bir LLM cevabına görüntü ekleyerek çıktıyı zenginleştirdik. Multimodal RAG çıktı zenginleştirmesi, hem girdi hem de çıktıya bilgi ekleyerek üretken yapay zekayı zenginleştirecektir. Ancak, tüm yapay zeka programlarında olduğu gibi, bir performans metriği tasarlamak verimli görüntü tanıma işlevselliği gerektirir.

## Kullanılan Kodların Tam Listesi (Complete Code List)
```python
import textwrap
from PIL import Image
from IPython.display import display

def display_source_image(image_path):
    image = Image.open(image_path)
    display(image)

# 1. Kullanıcı girdisi (user input)
user_input = user_input
print(user_input)

# 2. LLM cevabı (LLM response)
llm_response = llm_response
print(textwrap.fill(str(llm_response), 100))

# 3. Multimodal cevap (multimodal response)
image_path = "/content/boxed_image.jpg"
display_source_image(image_path)
```
Bu kod, kullanıcı girdisini, LLM çıktısını ve multimodal cevabı görüntülemek için kullanılır. `display_source_image` fonksiyonu, belirtilen yolundaki görüntüyü görüntüler. `textwrap.fill` fonksiyonu, LLM çıktısını biçimlendirmek için kullanılır.

---

## Performance metric

## Performans Metriği (Performance Metric)

Bir multimodal modüler RAG'ın performansını ölçmek iki tür ölçüm gerektirir: metin (text) ve görüntü (image). Metin ölçümü basit olsa da, görüntü ölçümü oldukça zordur. Multimodal bir yanıtın (response) görüntüsünü analiz etmek oldukça farklıdır. Multimodal sorgu motorundan (multimodal query engine) bir anahtar kelime (keyword) çıkardık. Daha sonra görüntülemek için bir kaynak görüntü (source image) için yanıtı ayrıştırdık (parsed the response). Ancak, yanıtın kaynak görüntüsünü değerlendirmek için yenilikçi bir yaklaşım oluşturmamız gerekecek. LLM performansıyla başlayalım.

## Önemli Noktalar
* Multimodal modüler RAG'ın performansını ölçmek için metin ve görüntü ölçümleri gereklidir.
* Metin ölçümü basittir, ancak görüntü ölçümü zordur.
* Multimodal yanıtın görüntüsünü analiz etmek farklı bir yaklaşıma ihtiyaç duyar.
* Multimodal sorgu motorundan anahtar kelime çıkarmak mümkündür.
* Yanıtın kaynak görüntüsünü değerlendirmek için yenilikçi bir yaklaşım gereklidir.

## Kod Örneği
Aşağıdaki kod örneği, multimodal sorgu motorundan anahtar kelime çıkarmak için kullanılabilir.
```python
import nltk
from nltk.tokenize import word_tokenize
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Multimodal sorgu motorundan anahtar kelime çıkarmak için kullanılan kod
def extract_keyword(query):
    tokens = word_tokenize(query)
    keyword = tokens[0]  # basit bir örnek, gerçek uygulamada daha karmaşık bir yaklaşım kullanılmalıdır
    return keyword

# Kaynak görüntü için yanıtı ayrıştırmak için kullanılan kod
def parse_response(response):
    # Yanıtı ayrıştırmak için kullanılan kod
    source_image = response["source_image"]
    return source_image

# CLIP modeli kullanarak görüntü ve metin arasındaki ilişkiyi değerlendirmek için kullanılan kod
def evaluate_image_text_relation(image, text):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    inputs = processor(text=text, images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    
    return logits_per_image

# Örnek kullanım
query = "örnek sorgu"
keyword = extract_keyword(query)
print("Anahtar Kelime:", keyword)

response = {"source_image": Image.open("image.jpg")}
source_image = parse_response(response)
print("Kaynak Görüntü:", source_image)

image_text_relation = evaluate_image_text_relation(source_image, keyword)
print("Görüntü-Metin İlişkisi:", image_text_relation)
```
## Kod Açıklaması

* `extract_keyword` fonksiyonu, multimodal sorgu motorundan anahtar kelime çıkarmak için kullanılır. Bu örnekte basit bir tokenization yaklaşımı kullanılmıştır, gerçek uygulamada daha karmaşık bir yaklaşım kullanılmalıdır.
* `parse_response` fonksiyonu, yanıtı ayrıştırmak ve kaynak görüntüyü elde etmek için kullanılır.
* `evaluate_image_text_relation` fonksiyonu, CLIP modeli kullanarak görüntü ve metin arasındaki ilişkiyi değerlendirmek için kullanılır. Bu kod, görüntü ve metin arasındaki benzerliği ölçmek için kullanılır.

## Notlar

* Kod örnekleri, basit birer örnektir ve gerçek uygulamada daha karmaşık ve detaylı bir şekilde kullanılmalıdır.
* CLIP modeli, görüntü ve metin arasındaki ilişkiyi değerlendirmek için kullanılan bir derin öğrenme modelidir.
* `transformers` kütüphanesini kullanarak CLIP modelini yüklemek ve kullanmak mümkündür.

---

## LLM performance metric

## LLM Performans Metriği (LLM Performance Metric)
LlamaIndex, sorgulama motoru (query engine) aracılığıyla GPT-4 gibi bir OpenAI modelini sorunsuz bir şekilde çağırır ve yanıtında metin içeriği sağlar. Metin yanıtları için, 2. Bölüm'deki "kosinüs benzerliği ile çıktı değerlendirme" bölümünde ve 3. Bölüm'deki "Vektör deposu indeks sorgulama motoru" bölümünde olduğu gibi aynı kosinüs benzerliği metriğini (cosine similarity metric) kullanacağız.

## Değerlendirme Fonksiyonu (Evaluation Function)
Değerlendirme fonksiyonu, iki metin arasındaki benzerliği değerlendirmek için `sklearn` ve `sentence_transformers` kütüphanelerini kullanır. Bu durumda, girdi ve çıktı arasındaki benzerliği değerlendirir.

### Kod
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity_with_embeddings(text1, text2):
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    similarity = cosine_similarity([embeddings1], [embeddings2])
    return similarity[0][0]
```
### Açıklama
- `SentenceTransformer` modeli, metinleri vektörlere dönüştürmek için kullanılır. Burada `'all-MiniLM-L6-v2'` modeli kullanılmıştır.
- `calculate_cosine_similarity_with_embeddings` fonksiyonu, iki metin arasındaki kosinüs benzerliğini hesaplar.
- `model.encode()` fonksiyonu, metinleri vektörlere dönüştürür.
- `cosine_similarity()` fonksiyonu, iki vektör arasındaki kosinüs benzerliğini hesaplar.

## Benzerlik Skorunun Hesaplanması (Calculating Similarity Score)
Şimdi, temel kullanıcı girdisi (baseline user input) ve elde edilen ilk LLM yanıtı arasındaki benzerliği hesaplayabiliriz.

### Kod
```python
llm_similarity_score = calculate_cosine_similarity_with_embeddings(user_input, str(llm_response))
print(user_input)
print(llm_response)
print(f"Cosine Similarity Score: {llm_similarity_score:.3f}")
```
### Açıklama
- `user_input` ve `llm_response` değişkenleri sırasıyla kullanıcı girdisini ve LLM yanıtını temsil eder.
- `calculate_cosine_similarity_with_embeddings` fonksiyonu, bu iki metin arasındaki kosinüs benzerliğini hesaplar.
- Sonuç, `.3f` formatında yazılır, yani üç ondalık basamağa yuvarlanır.

## Çıktı (Output)
Çıktı, kullanıcı girdisini, metin yanıtını ve iki metin arasındaki kosinüs benzerliğini gösterir.

### Örnek Çıktı
```
How do drones identify a truck?
Drones can identify a truck using visual detection and tracking methods, which may involve deep neural networks for performance benchmarking.
Cosine Similarity Score: 0.691
```
## Multimodal Performansın Ölçülmesi (Measuring Multimodal Performance)
Çıktı tatmin edicidir. Ancak şimdi, multimodal performansı ölçmek için bir yol tasarlamamız gerekiyor.

---

## Multimodal performance metric

## Multimodal Performans Metriği (Multimodal Performance Metric)

Geri dönen görüntüyü değerlendirmek için basitçe veri kümesindeki etiketlere güvenemeyiz. Küçük veri kümeleri için görüntüyü manuel olarak kontrol edebiliriz, ancak bir sistem ölçeklendiğinde otomasyon gereklidir. Bu bölümde, GPT-4o'nun (GPT-4o) bilgisayarlı görü özelliklerini kullanarak bir görüntüyü analiz edeceğiz, aradığımız nesneleri bulacağız ve bu görüntünün bir açıklamasını sağlayacağız. Ardından, GPT-4o tarafından sağlanan açıklama ile içermesi gereken etiket arasındaki kosinüs benzerliğini (Cosine Similarity) uygulayacağız.

## GPT-4o ve Base64 Kodlama (GPT-4o and Base64 Encoding)

GPT-4o, multimodal bir üretken yapay zeka modelidir (Multimodal Generative AI Model). İlk olarak, veri iletimini GPT-4o'ya basitleştirmek için görüntüyü kodlayacağız. Base64 kodlama, ikili verileri (görüntüler gibi) standart metin karakterleri olan ASCII karakterlerine dönüştürür. Bu dönüşüm, görüntü verilerinin metin verilerini sorunsuz bir şekilde işlemek üzere tasarlanmış protokoller (HTTP gibi) üzerinden iletilmesini sağlar. Ayrıca, veri bozulması veya yorumlama hataları gibi ikili veri iletimiyle ilgili sorunları önler.

### Kodlama İşlemi (Encoding Process)

```python
import base64

IMAGE_PATH = "/content/boxed_image.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image(IMAGE_PATH)
```

Bu kod, `/content/boxed_image.jpg` yolundaki görüntüyü Base64 formatına kodlar. Kodlama işlemi, `base64` kütüphanesi kullanılarak yapılır. Kodlanan görüntü, `base64_image` değişkenine atanır.

## OpenAI İstemcisi ve GPT-4o Modeli (OpenAI Client and GPT-4o Model)

OpenAI istemcisini oluşturacağız ve modeli `gpt-4o` olarak ayarlayacağız:

```python
from openai import OpenAI

client = OpenAI(api_key=openai.api_key)
MODEL = "gpt-4o"
```

Bu kod, OpenAI API anahtarını kullanarak bir istemci oluşturur ve modeli `gpt-4o` olarak ayarlar.

## Görüntü Analizi ve Kosinüs Benzerliği (Image Analysis and Cosine Similarity)

Görüntüyü GPT-4o'ya göndereceğiz ve analiz sonucunu alacağız:

```python
u_word = unique_words[0]
print(u_word)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": f"You are a helpful assistant that analyzes images that contain {u_word}."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Analyze the following image, tell me if there is one {u_word} or more in the bounding boxes and analyze them:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    temperature=0.0,
)

response_image = response.choices[0].message.content
print(response_image)
```

Bu kod, `u_word` değişkenindeki etiketi içeren görüntüyü analiz etmek için GPT-4o'ya bir istek gönderir. Yanıt, `response_image` değişkenine atanır.

## Kosinüs Benzerliği Hesaplanması (Calculating Cosine Similarity)

Analiz sonucunu kosinüs benzerliği fonksiyonuna göndereceğiz:

```python
resp = u_word + "s"
multimodal_similarity_score = calculate_cosine_similarity_with_embeddings(resp, str(response_image))
print(f"Cosine Similarity Score: {multimodal_similarity_score:.3f}")
```

Bu kod, analiz sonucunu ve etiketi kosinüs benzerliği fonksiyonuna gönderir ve benzerlik skorunu hesaplar.

## Önemli Noktalar (Key Points)

*   Multimodal performans metriği, görüntü analizi ve kosinüs benzerliği kullanılarak hesaplanır.
*   GPT-4o, multimodal bir üretken yapay zeka modelidir.
*   Base64 kodlama, ikili verileri ASCII karakterlerine dönüştürür.
*   OpenAI istemcisi ve GPT-4o modeli, görüntü analizi için kullanılır.
*   Kosinüs benzerliği, analiz sonucu ve etiket arasındaki benzerliği ölçer.

## Kullanılan Kodlar ve Açıklamaları (Used Codes and Explanations)

*   `encode_image` fonksiyonu, görüntüyü Base64 formatına kodlar.
*   `OpenAI` istemcisi, OpenAI API anahtarını kullanarak oluşturulur.
*   `client.chat.completions.create` fonksiyonu, GPT-4o'ya görüntü analizi için bir istek gönderir.
*   `calculate_cosine_similarity_with_embeddings` fonksiyonu, kosinüs benzerliğini hesaplar.

---

## Multimodal modular RAG performance metric

## Multimodal Modüler RAG Performans Metriği (Multimodal Modular RAG Performance Metric)

Sistemin genel performansını elde etmek için, LLM (Large Language Model) yanıtının ve iki multimodal yanıt performansının toplamını 2'ye böleceğiz.

## Performans Hesaplaması

```python
score = (llm_similarity_score + multimodal_similarity_score) / 2
print(f"Multimodal, Modüler Skor: {score:.3f}")
```

Bu kod, `llm_similarity_score` ve `multimodal_similarity_score` değişkenlerini kullanarak sistemin genel performansını hesaplar ve sonucu `score` değişkenine atar. Daha sonra sonucu ekrana yazdırır.

- `llm_similarity_score`: LLM modelinin benzerlik skoru
- `multimodal_similarity_score`: Multimodal modelin benzerlik skoru

Bu kodun altında, `print` fonksiyonu kullanılarak sonuç ekrana yazdırılır. `f-string` formatı kullanılarak, `score` değişkeninin değeri `.3f` formatında yazdırılır, yani virgülden sonra 3 basamak gösterilir.

## Sonuç

Sonuç, bir insan gözlemcinin sonuçlardan memnun olabileceğini, ancak karmaşık bir görüntünün uygunluğunu otomatik olarak değerlendirmek zor olduğunu gösterir.

## Multimodal Modüler Skor

Multimodal, Modüler Skor: 0.598

Bu metrik, bir insan gözlemcinin görüntünün uygun olduğunu görmesi nedeniyle geliştirilebilir. Bu, ChatGPT, Gemini ve Bing Copilot gibi üst düzey AI ajanlarının neden her zaman başparmak yukarı ve başparmak aşağı içeren bir geri bildirim süreci olduğunu açıklar.

## İyileştirme

Bu metrik, insan geri bildirimi ile daha da geliştirilebilir. Bu, RAG'ın (Retrieval-Augmented Generator) daha da geliştirilmesi için insan geri bildiriminin nasıl kullanılabileceğini keşfetmek için bir sonraki adımdır.

## Özet

Bu bölüm, multimodal modüler RAG performans metriğini ve sistemin genel performansını hesaplamak için kullanılan kodu açıklar. Ayrıca, bu metriğin geliştirilmesi ve RAG'ın daha da iyileştirilmesi için insan geri bildiriminin nasıl kullanılabileceği hakkında bilgi verir.

Gerekli import kütüphaneleri bu kod için belirtilmemiştir, ancak genel olarak benzerlik skorlarını hesaplamak için kullanılan kütüphaneler arasında `transformers`, `torch` ve `numpy` sayılabilir.

```python
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
```

Bu kütüphaneler, doğal dil işleme ve derin öğrenme görevleri için kullanılır. `transformers` kütüphanesi, önceden eğitilmiş dil modellerini yüklemek ve kullanmak için kullanılır. `torch` kütüphanesi, derin öğrenme modellerini oluşturmak ve eğitmek için kullanılır. `numpy` kütüphanesi, sayısal işlemler için kullanılır.

---

## Summary

## Çok Modlu Modüler RAG (Multimodal Modular RAG) Sistemine Giriş
Bu bölüm, farklı veri türleri (metin ve görüntü) ve görevler için ayrı modüller kullanan çok modlu modüler RAG (Multimodal Modular RAG) dünyasına giriş yaptı. Daha önceki bölümlerde keşfettiğimiz LlamaIndex, Deep Lake ve OpenAI'ın işlevselliğinden yararlandık.

## Kullanılan Teknolojiler ve Veri Kümeleri
- **LlamaIndex**: Veri indeksleme ve sorgulama için kullanıldı.
- **Deep Lake**: Veri depolama ve yönetme için kullanıldı. 
  - **VisDrone Veri Kümesi**: İHA (İnsansız Hava Aracı) teknolojisi için görüntü analizi ve nesne tanıma amacıyla kullanıldı. Bu veri kümesi görüntüler, etiketler ve sınırlayıcı kutu (bounding box) bilgileri içeriyordu.

## Çok Modlu Modüler RAG Sisteminin Geliştirilmesi
1. **Temel Kullanıcı Sorgusunun Tanımlanması**: Hem LLM (Large Language Model) hem de çok modlu sorgular için temel bir kullanıcı sorgusu tanımlandı.
2. **Metin Veri Kümesinin Sorgulanması**: Bölüm 3'te uygulanan Deep Lake metin veri kümesi sorgulandı. LlamaIndex, bir sorgu motorunu sorunsuz bir şekilde çalıştırarak yanıtı aldı, genişletti ve oluşturdu.
3. **Görüntü Veri Kümesinin İşlenmesi**: Deep Lake VisDrone veri kümesi yüklendi ve LlamaIndex ile bellekte indekslenerek bir indeksli vektör arama alma hattı (indexed vector search retrieval pipeline) oluşturuldu.
4. **Görüntü ve Metin Yanıtlarının Birleştirilmesi**: LlamaIndex aracılığıyla GPT-4 gibi bir OpenAI modeli kullanılarak sorgulandı, oluşturulan metin bir anahtar kelime için ayrıştırıldı. Kaynak düğümleri (source nodes) aranarak kaynak görüntü bulundu, görüntülendi ve LLM ile görüntü yanıtları genişletilmiş bir çıktı oluşturmak üzere birleştirildi.

## Değerlendirme ve Benzerlik Ölçümü
- **Metin Yanıtı İçin Kosinüs Benzerliği**: Metin yanıtına kosinüs benzerliği uygulandı.
- **Görüntü Değerlendirmesi**: Görüntünün değerlendirilmesi zor olduğundan, önce GPT-4o ile görüntü tanıma çalıştırılarak bir metin elde edildi ve bu metne kosinüs benzerliği uygulandı.

## Kod Örneği ve Açıklaması
Aşağıdaki kod örneği, LlamaIndex ve Deep Lake kullanarak çok modlu modüler RAG sisteminin nasıl geliştirileceğini gösterir:
```python
# Import gerekli kütüphaneler
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.storage.storage_context import StorageContext
import deeplake
from PIL import Image
import openai

# Deep Lake veri kümesini yükleme
ds = deeplake.load('hub://activeloop/visdrone')

# LlamaIndex ile indeks oluşturma
storage_context = StorageContext.from_defaults(vector_store=DeepLakeVectorStore(dataset_path="hub://activeloop/visdrone"))
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Sorgu motorunu çalıştırma
query_engine = index.as_query_engine()
response = query_engine.query("Sorgu metni")

# Kaynak görüntüleri bulma ve görüntüleme
for node in response.source_nodes:
    image_path = node.metadata['image_path']
    image = Image.open(image_path)
    image.show()

# Kosinüs benzerliği hesaplama
from sklearn.metrics.pairwise import cosine_similarity
# Metin vektörleri
text_vector1 = [0.1, 0.2, 0.3]
text_vector2 = [0.4, 0.5, 0.6]
cosine_similarity([text_vector1], [text_vector2])
```
## Kodun Açıklaması
- İlk olarak, gerekli kütüphaneler içe aktarılır (`llama_index`, `deeplake`, `PIL`, `openai`).
- Deep Lake veri kümesi (`visdrone`) yüklenir.
- LlamaIndex kullanarak bir indeks oluşturulur. Bu indeks, veri kümesindeki belgeleri (`documents`) saklar.
- Bir sorgu motoru (`query_engine`) oluşturulur ve bir sorgu (`"Sorgu metni"`) çalıştırılır.
- Yanıttaki kaynak düğümleri (`source_nodes`) kullanılarak kaynak görüntüler bulunur ve görüntülenir.
- Kosinüs benzerliği (`cosine_similarity`), iki metin vektörü arasındaki benzerliği ölçmek için kullanılır.

## Önemli Noktalar
- **Çok Modlu Modüler RAG**: Farklı veri türleri ve görevler için ayrı modüller kullanır.
- **LlamaIndex ve Deep Lake**: Veri indeksleme, sorgulama ve depolama için kullanılır.
- **Görüntü ve Metin İşleme**: Hem görüntü hem de metin verileri işlenir ve birleştirilir.
- **Kosinüs Benzerliği**: Metin yanıtlarının benzerliğini ölçmek için kullanılır.
- **Gelecek Adımlar**: Bir sonraki bölüm, şeffaflık ve kesinlik konularına daha derinlemesine girecektir.

---

## Questions

## Sorular ve Cevaplar
Aşağıdaki paragrafta anlatılan konuyu türkçe olarak tekrar düzenleyerek önemli noktaları maddeler halinde yazacağım. Ayrıca, text içinde kodlar varsa yazıp açıklayacağım ve türkçenin yanına ingilizce teknik terimleri parantez içinde ekleyeceğim.

## Konu
Çok modlu modüler RAG (Multimodal Modular RAG) sisteminin farklı veri türlerini işleyip işlemediği, drone (İHA - İnsansız Hava Aracı) kullanım alanları, Deep Lake VisDrone veri setinin kullanım amacı, bounding box (sınırlayıcı kutu) ekleme ve sistemin performansı gibi konular ele alınmaktadır.

## Önemli Noktalar
* Çok modlu modüler RAG (Multimodal Modular RAG) farklı veri türlerini işleyebilir (text ve görüntüler).
* Drone (İHA) yalnızca tarım izleme ve hava fotoğrafçılığı için kullanılmaz.
* Deep Lake VisDrone veri seti bu bölümde yalnızca metinsel veriler için kullanılmamıştır.
* Drone görüntüleri üzerine bounding box (sınırlayıcı kutu) eklenerek nesneler (örneğin kamyon ve yayalar) tanımlanabilir.
* Modüler sistem sorgu yanıtları için hem metin hem de görüntü verilerini alır.
* Çok modlu VisDrone veri setini sorgulamak için vektör indeksi (vector index) oluşturmak gereklidir.
* Alınan görüntüler etiket veya bounding box eklenmeden işlenmez.
* Çok modlu modüler RAG performans metriği (performance metric) yalnızca metinsel yanıtlara dayanmaz.
* Bu bölümde anlatılan çok modlu sistem yalnızca drone ile ilgili verileri işleyemez.
* Çok modlu RAG'de görüntüleri değerlendirmek metinleri değerlendirmek kadar kolay değildir.

## Sorular ve Cevaplar
1. Çok modlu modüler RAG (Multimodal Modular RAG) metin ve görüntüler gibi farklı veri türlerini işleyebilir mi? 
## Evet (Yes)
2. Drone (İHA) yalnızca tarım izleme ve hava fotoğrafçılığı için kullanılır mı? 
## Hayır (No)
3. Deep Lake VisDrone veri seti bu bölümde yalnızca metinsel veriler için mi kullanılmıştır? 
## Hayır (No)
4. Drone görüntüleri üzerine nesneleri (örneğin kamyon ve yayalar) tanımlamak için bounding box (sınırlayıcı kutu) eklenebilir mi? 
## Evet (Yes)
5. Modüler sistem sorgu yanıtları için hem metin hem de görüntü verilerini alır mı? 
## Evet (Yes)
6. Çok modlu VisDrone veri setini sorgulamak için vektör indeksi (vector index) oluşturmak gerekli midir? 
## Evet (Yes)
7. Alınan görüntüler etiket veya bounding box eklenmeden işlenir mi? 
## Hayır (No)
8. Çok modlu modüler RAG performans metriği (performance metric) yalnızca metinsel yanıtlara mı dayanır? 
## Hayır (No)
9. Bu bölümde anlatılan çok modlu sistem yalnızca drone ile ilgili verileri işleyebilir mi? 
## Hayır (No)
10. Çok modlu RAG'de görüntüleri değerlendirmek metinleri değerlendirmek kadar kolay mıdır? 
## Hayır (No)

## Kodlar ve Açıklamaları
Bu bölümde kullanılan kodlara örnekler verilecek ve bu kodların nasıl kullanıldığı açıklanacaktır. Ancak, orijinal metinde kod örneği bulunmadığından, çok modlu modüler RAG sisteminin temel bileşenlerini göstermek için örnek bir kod yapısı aşağıda verilmiştir.

```python
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

## CLIP model ve processor'ünü yükleme (Loading CLIP model and processor)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

## Görüntü ve metin verilerini işleme (Processing image and text data)
def process_data(image_path, text):
    image = Image.open(image_path)
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    return inputs

## CLIP model ile sorgu yapma (Querying with CLIP model)
def query_clip(inputs):
    outputs = model(**inputs)
    return outputs

## Örnek kullanım (Example usage)
image_path = "path/to/image.jpg"
text = ["A drone image with trucks and pedestrians."]
inputs = process_data(image_path, text)
outputs = query_clip(inputs)

## Sonuçları işleme (Processing results)
# Burada, alınan çıktıların nasıl işleneceği gösterilmektedir.
# Örneğin, benzerlik skorları hesaplanabilir veya bounding box'lar eklenebilir.
```

Bu kod örneğinde, CLIP (Contrastive Language-Image Pre-training) model kullanılarak görüntü ve metin verileri işlenmektedir. CLIP, görüntü ve metin arasındaki ilişkiyi anlamak için kullanılan bir modeldir. Kod, görüntü ve metin girdi olarak alıp, bu girdileri CLIP model ile işleyerek çıktı üretmektedir.

## Kodun Kullanımı
1. Gerekli kütüphaneleri (`numpy`, `torch`, `PIL`, `transformers`) yükleyin.
2. CLIP model ve processor'ünü yüklemek için `CLIPModel.from_pretrained` ve `CLIPProcessor.from_pretrained` metodlarını kullanın.
3. `process_data` fonksiyonu ile görüntü ve metin verilerini işleyin.
4. `query_clip` fonksiyonu ile CLIP model kullanarak sorgu yapın.
5. Alınan çıktıları işleyerek, örneğin benzerlik skorları hesaplayarak veya bounding box'lar ekleyerek sonuçları değerlendirin.

Bu örnek, çok modlu modüler RAG sisteminin temel bileşenlerini göstermek amacıyla hazırlanmıştır. Gerçek sistemde, daha karmaşık işlemler ve farklı model veya algoritmalar kullanılabilir.

---

## References

## LlamaIndex, Activeloop Deep Lake ve OpenAI Hakkında Genel Bakış
LlamaIndex, Activeloop Deep Lake ve OpenAI, yapay zeka ve makine öğrenimi alanında kullanılan önemli araçlardır. Bu araçlar, geliştiricilere ve araştırmacılara veri işleme, depolama ve analizinde güçlü çözümler sunar.

## LlamaIndex
LlamaIndex, büyük veri kümeleri üzerinde indeksleme ve sorgulama işlemlerini kolaylaştıran bir araçtır. Bu, büyük veri kümeleri üzerinde hızlı ve verimli sorgulama yapma imkanı sağlar.

### Önemli Noktalar:
- **Hızlı Sorgulama**: LlamaIndex, büyük veri kümeleri üzerinde hızlı sorgulama yapma imkanı sağlar.
- **Veri İndeksleme**: Veri kümelerini indeksler ve bu sayede daha hızlı erişim sağlar.

## Activeloop Deep Lake
Activeloop Deep Lake, büyük ölçekli veri kümelerini depolamak ve yönetmek için kullanılan bir veri gölü çözümüdür. Bu, veri bilimcileri ve makine öğrenimi mühendisleri için veri yönetimini kolaylaştırır.

### Önemli Noktalar:
- **Veri Depolama**: Büyük ölçekli veri kümelerini depolama imkanı sağlar.
- **Veri Yönetimi**: Veri kümelerini yönetmek için güçlü araçlar sunar.

## OpenAI
OpenAI, yapay zeka modelleri geliştiren ve bu modelleri geliştiricilerin kullanımına sunan bir organizasyondur. OpenAI modelleri, doğal dil işleme ve diğer yapay zeka görevlerinde kullanılır.

### Önemli Noktalar:
- **Yapay Zeka Modelleri**: Gelişmiş yapay zeka modelleri sağlar.
- **Doğal Dil İşleme**: Doğal dil işleme görevlerinde kullanılan modeller sunar.

## Kod Örnekleri ve Kullanımları
Aşağıda, bu araçlarla ilgili bazı kod örnekleri verilmiştir.

### LlamaIndex Kullanımı
```python
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex

# Veri yükleme
documents = SimpleDirectoryReader('data').load_data()

# İndeks oluşturma
index = GPTSimpleVectorIndex(documents)

# Sorgulama yapma
response = index.query("Sorgunuzu buraya yazın.")
print(response)
```
**Kod Açıklaması**: Bu kod, `SimpleDirectoryReader` kullanarak bir dizindeki verileri yükler, `GPTSimpleVectorIndex` ile bir indeks oluşturur ve daha sonra bu indeks üzerinde bir sorgulama yapar.

### Activeloop Deep Lake Kullanımı
```python
import deeplake

# Veri seti oluşturma
ds = deeplake.dataset('hub://username/dataset')

# Veri ekleme
ds.append({'image': deeplake.link('https://example.com/image.jpg'), 'label': 1})

# Veri okuma
print(ds['image'][0].numpy().shape)
```
**Kod Açıklaması**: Bu kod, Activeloop Deep Lake kullanarak bir veri seti oluşturur, bu veri setine veri ekler ve daha sonra bu veriyi okur.

### OpenAI Kullanımı
```python
import openai

# API anahtarını ayarla
openai.api_key = "API_ANAHTARINIZ"

# Modeli çağırma
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Merhaba, nasılsınız?",
  max_tokens=1024
)

# Yanıtı yazdırma
print(response.choices[0].text.strip())
```
**Kod Açıklaması**: Bu kod, OpenAI API'sini kullanarak bir metin tamamlama görevi gerçekleştirir. `openai.Completion.create` methodu ile bir istek gönderir ve gelen yanıtı işler.

## Kaynaklar
- [LlamaIndex Dokümantasyonu](https://docs.llamaindex.ai/en/stable/)
- [Activeloop Deep Lake Dokümantasyonu](https://docs.activeloop.ai/)
- [OpenAI Dokümantasyonu](https://platform.openai.com/docs/overview)

---

## Further reading

## Daha Fazla Okuma (Further Reading)
Retrieval-Augmented Multimodal Language Modeling (Çok Modlu Dil Modellemesinde Erişim-Destekli Yaklaşım) konusunda Yasunaga ve ekibi tarafından 2023 yılında yayınlanan makale, bu alandaki önemli gelişmeleri ve teknikleri ele almaktadır. Makalenin içeriği ve öne çıkan noktalar aşağıda özetlenmiştir.

## Önemli Noktalar (Key Points)
- **Çok Modlu Dil Modelleme (Multimodal Language Modeling)**: Metin, resim, ses gibi farklı veri türlerini bir arada kullanarak dil modelleme yapmak.
- **Erişim-Destekli Yaklaşım (Retrieval-Augmented Approach)**: Modelin performansı ve bilgisi, dış kaynaklardan erişilen bilgilerle desteklenir.
- **Gelişmiş Model Performansı (Enhanced Model Performance)**: Erişim-destekli yaklaşım, modelin daha doğru ve zengin çıktılar üretmesini sağlar.

## Teknik Detaylar (Technical Details)
Makale, erişim-destekli çok modlu dil modelleme konusunda çeşitli teknik detayları içerir. Bu teknik detaylar arasında:
- **Veri Erişim Yöntemleri (Data Retrieval Methods)**: İlgili bilgilerin dış kaynaklardan nasıl erişileceği ve entegre edileceği.
- **Model Mimari Tasarımları (Model Architecture Designs)**: Çok modlu girdileri işleyebilen ve erişilen bilgileri model içine entegre edebilen mimari tasarımlar.

## Kod Örnekleri (Code Examples)
Aşağıda, makalede bahsi geçen bazı temel kod örnekleri ve açıklamaları verilmiştir. Bu örnekler, Python programlama dili ve popüler derin öğrenme kütüphaneleri kullanılarak yazılmıştır.

### Örnek Kod 1: Veri Erişim Fonksiyonu (Data Retrieval Function)
```python
import requests
from bs4 import BeautifulSoup

def retrieve_data(query):
    url = f"https://example.com/search?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # İlgili verileri parse et
    data = soup.find_all('div', {'class': 'result'})
    return [item.text.strip() for item in data]

# Kullanım örneği
query = "örnek sorgu"
results = retrieve_data(query)
print(results)
```
Bu kod, belirli bir sorguya (`query`) göre dış kaynaktan (`https://example.com/search`) veri erişimi yapar (`requests.get(url)`). Erişilen sayfa içeriği (`response.content`), `BeautifulSoup` kütüphanesi kullanılarak parse edilir ve ilgili veriler (`data`) çekilir.

### Örnek Kod 2: Çok Modlu Dil Modelleme (Multimodal Language Modeling)
```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Model ve tokenizer yükle
model_name = "example-multimodal-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_text(prompt, image_features):
    inputs = tokenizer(prompt, return_tensors="pt")
    # Görüntü özelliklerini modele entegre et
    outputs = model.generate(**inputs, image_features=image_features)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Kullanım örneği
prompt = "örnek metin"
image_features = torch.randn(1, 512)  # Örnek görüntü özellikleri
generated_text = generate_text(prompt, image_features)
print(generated_text)
```
Bu örnek, önceden eğitilmiş (`pre-trained`) bir çok modlu dil modelini (`AutoModelForSeq2SeqLM`) kullanarak metin üretimini (`generate_text`) gösterir. Model, hem metin girdisini (`prompt`) hem de görüntü özelliklerini (`image_features`) işler.

## Kaynaklar (References)
- Yasunaga et al. (2023), "Retrieval-Augmented Multimodal Language Modeling", https://arxiv.org/pdf/2211.12561

## Eklemeler (Additional Notes)
- Erişim-destekli çok modlu dil modelleme, çeşitli uygulama alanlarında (örneğin, görsel soru-cevap, metin-görüntü sentezleme) kullanılabilir.
- Bu teknik, büyük dil modellerinin (`Large Language Models`) yeteneklerini daha da genişletme potansiyelini taşır.

---

## Join our community on Discord

## Discord Topluluğumuza Katılın (Join our community on Discord)
Packt linki üzerinden Discord platformunda yazar ve diğer okuyucularla tartışmalara katılmak için topluluğumuza katılabilirsiniz.

## Discord Topluluğumuzun Önemi
Discord topluluğumuza katılarak, yazar ve diğer okuyucularla tartışmalara katılabilir, bilgi ve deneyimlerinizi paylaşabilirsiniz.

## Discord'a Katılma Adresi
Discord topluluğumuza katılmak için aşağıdaki bağlantıyı kullanabilirsiniz:
https://www.packt.link/rag

## Bağlantının Açıklaması
Bağlantı, sizi Packt'ın Discord sunucusuna yönlendirecektir. Burada, yazar ve diğer okuyucularla sohbet edebilir, sorularınızı sorabilir ve bilgi paylaşabilirsiniz.

## Kod Örneği Yok
Bu metinde herhangi bir kod örneği bulunmamaktadır.

## Önemli Noktalar
* Discord topluluğumuza katılmak için Packt linkini kullanın.
* Yazar ve diğer okuyucularla tartışmalara katılın.
* Bilgi ve deneyimlerinizi paylaşın.

## Teknik Terimler
* Discord (Discord): bir iletişim platformu.
* Packt (Packt): bir yayıncılık şirketi.

## Ek Bilgiler
Discord, geliştiriciler ve diğer profesyoneller arasında iletişimi kolaylaştırmak için tasarlanmış bir platformdur. Packt ise, teknoloji ve yazılım konularında kitaplar ve diğer kaynaklar yayınlayan bir şirketidir.

---

