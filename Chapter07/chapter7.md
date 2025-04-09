## Building Scalable Knowledge-Graph-Based RAG with Wikipedia API and LlamaIndex

## Büyük Ölçekli Bilgi Grafiği Tabanlı RAG (Retrieval-Augmented Generation) Sistemi Oluşturma: Wikipedia API ve LlamaIndex ile

Büyük veri setleri (large datasets) hızla yönetilmesi zor hale gelebilir. Gerçek hayattaki projelerde, veri yönetimi (data management) yapay zeka (AI) dan daha fazla baş ağrısı yaratır! Proje yöneticileri (project managers), danışmanlar (consultants) ve geliştiriciler (developers) gerekli verileri elde etmek için sürekli mücadele ederler. Veri genellikle yapılandırılmamış (unstructured) bir şekilde başlar ve daha sonra ağrılı karar verme süreçleri (painful decision-making processes) yoluyla bir şekilde organize edilir. Wikipedia, veri ölçeklendirilmesinin (scaling data) çoğunlukla güvenilir ancak bazen yanlış bilgilere yol açmasının iyi bir örneğidir. Gerçek hayattaki projeler genellikle Wikipedia'nın yaptığı gibi gelişir. Veri bir şirket içinde sürekli olarak birikmeye devam eder ve veritabanı yöneticilerini (database administrators), proje yöneticilerini (project managers) ve kullanıcıları (users) zorlar.

## Bilgi Grafikleri (Knowledge Graphs) ile Büyük Veri Yönetimi

Büyük miktarda verinin nasıl bir araya geldiğini görmek ana sorunlardan biridir ve bilgi grafikleri (knowledge graphs) farklı veri türleri arasındaki ilişkileri görselleştirmenin (visualizing) etkili bir yolunu sağlar. Bu bölüm, RAG tabanlı üretken yapay zeka (RAG-driven generative AI) için tasarlanmış bir bilgi tabanı ekosisteminin (knowledge base ecosystem) mimarisini tanımlayarak başlar. Ekosistem üç işlem hattından (pipelines) oluşur: veri toplama (data collection), bir vektör deposu (vector store) doldurma ve bilgi grafiği indeksi tabanlı (knowledge graph index-based) bir RAG programı çalıştırma.

### İşlem Hattı 1: Dokümanları Toplama ve Hazırlama

İlk adım, Wikipedia API ile otomatik bir Wikipedia alma programı oluşturmaktır. Bir Wikipedia sayfasına dayalı bir konu seçeceğiz ve ardından programın gerekli meta verileri toplamasına ve verileri hazırlamasına izin vereceğiz. Sistem esnek olacak ve istediğiniz herhangi bir konuyu seçmenize izin verecektir.

#### Kod:
```python
import wikipedia

def get_wikipedia_data(topic):
    try:
        page = wikipedia.page(topic)
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error: {e}")
        return None

topic = "Marketing"
data = get_wikipedia_data(topic)
print(data)
```
Bu kod, Wikipedia API'sini kullanarak belirli bir konu hakkında veri çeker. `wikipedia.page()` fonksiyonu, belirtilen konu hakkında bir Wikipedia sayfası döndürür ve `page.content` özelliği sayfanın içeriğini sağlar.

### İşlem Hattı 2: Deep Lake Vektör Deposu Oluşturma ve Doldurma

İkinci adım, verileri bir vektör deposuna (vector store) yüklemektir. Deep Lake'in otomatik parçalama (automated chunking) ve OpenAI embedding işlevselliğini kullanacağız.

#### Kod:
```python
import deeplake
from deeplake import Dataset

# Deep Lake dataset oluşturma
ds = Dataset("marketing_data")

# Verileri yükleme
ds.add(data)

# Otomatik parçalama ve embedding
ds.create_embedding(embedding_function="openai")
```
Bu kod, Deep Lake kullanarak bir vektör deposu oluşturur ve verileri yükler. `ds.create_embedding()` fonksiyonu, OpenAI embedding işlevselliğini kullanarak verileri gömülü (embedded) hale getirir.

### İşlem Hattı 3: Bilgi Grafiği İndeksi Tabanlı RAG

Üçüncü adım, LlamaIndex kullanarak bir bilgi grafiği indeksi oluşturmaktır. LlamaIndex, verilerdeki anlamsal ilişkileri (semantic relationships) gösteren bir grafik üretecektir.

#### Kod:
```python
import llama_index

# LlamaIndex oluşturma
index = llama_index.Index()

# Verileri ekleme
index.add(data)

# Bilgi grafiği indeksi oluşturma
index.create_knowledge_graph()
```
Bu kod, LlamaIndex kullanarak bir bilgi grafiği indeksi oluşturur. `index.create_knowledge_graph()` fonksiyonu, verilerdeki anlamsal ilişkileri gösteren bir grafik üretir.

## Sonuç

Bu bölümde, RAG tabanlı üretken yapay zeka için bir bilgi tabanı ekosisteminin mimarisini tanımladık. Üç işlem hattı oluşturduk: veri toplama, vektör deposu doldurma ve bilgi grafiği indeksi tabanlı RAG. Wikipedia API, Deep Lake ve LlamaIndex kullanarak büyük ölçekli veri yönetimini gösterdik.

---

## The architecture of RAG for knowledge-graph-based semantic search

## Bilgi Grafiği Tabanlı Anlamsal Arama için RAG Mimarisinin Oluşturulması (Building RAG Architecture for Knowledge-Graph-Based Semantic Search)

Bu bölümde, bir bilgi grafiği (knowledge graph) tabanlı RAG (Retrieval-Augmented Generation) programı oluşturacağız. Bilgi grafiği, bir RAG veri kümesindeki belgeler arasındaki ilişkileri görsel olarak haritalamamızı sağlar. Bu grafiği, LlamaIndex kullanarak otomatik olarak oluşturacağız.

## İşlem Adımları (Pipelines)

1. **Pipeline 1: Dokümanları Toplama ve Hazırlama (Collecting and Preparing Documents)**
   - Wikipedia API kullanarak bir Wikipedia konusuna ilişkin bağlantıları ve meta verileri (özet, URL ve alıntı verileri) alacağız.
   - Alınan URL'leri yükleyip ayrıştırmak suretiyle verileri hazırlayacağız.

2. **Pipeline 2: Deep Lake Vektör Deposu Oluşturma ve Doldurma (Creating and Populating Deep Lake Vector Store)**
   - Pipeline 1'de hazırlanan Wikipedia sayfalarının içeriğini gömerek (embedding) Deep Lake vektör deposuna yükleyeceğiz.

3. **Pipeline 3: Bilgi Grafiği Tabanlı RAG (Knowledge Graph Index-Based RAG)**
   - LlamaIndex kullanarak bilgi grafiği indeksini oluşturacağız ve bunu görüntüleyeceğiz.
   - Bilgi grafiği indeksini sorgulamak için işlevsellik oluşturacağız ve LlamaIndex'in dahili LLM (Large Language Model) özelliğini kullanarak sorguya yanıt üreteceğiz.

## Kod Parçaları ve Açıklamaları

### Wikipedia API ile Veri Alma

Wikipedia API kullanarak veri almak için aşağıdaki kod parçasını kullanacağız:
```python
import wikipedia

def get_wikipedia_data(topic):
    try:
        page = wikipedia.page(topic)
        return page.content, page.url, page.links
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error: {e}")
        return None, None, None

topic = "Marketing"
content, url, links = get_wikipedia_data(topic)
print(content)
print(url)
print(links)
```
Bu kod, belirtilen konu için Wikipedia sayfasının içeriğini, URL'sini ve bağlantılarını alır.

### LlamaIndex ile Bilgi Grafiği Oluşturma

LlamaIndex kullanarak bilgi grafiği oluşturmak için aşağıdaki kod parçasını kullanacağız:
```python
from llama_index import KnowledgeGraphIndex

def create_knowledge_graph_index(nodes):
    index = KnowledgeGraphIndex(nodes)
    return index

# nodes: Pipeline 1 ve Pipeline 2'dan gelen veriler
index = create_knowledge_graph_index(nodes)
```
Bu kod, belirtilen düğümler (nodes) için bir bilgi grafiği indeksi oluşturur.

### Sorgulama ve Yanıt Üretme

Bilgi grafiği indeksini sorgulamak ve yanıt üretmek için aşağıdaki kod parçasını kullanacağız:
```python
from llama_index import LLMPredictor

def query_knowledge_graph_index(index, query):
    llm_predictor = LLMPredictor()
    response = llm_predictor.query(index, query)
    return response

query = "What is marketing?"
response = query_knowledge_graph_index(index, query)
print(response)
```
Bu kod, belirtilen sorgu için bilgi grafiği indeksini sorgular ve bir yanıt üretir.

## Sonuç

Bu bölümde, bilgi grafiği tabanlı bir RAG programı oluşturduk. Bu program, Wikipedia API kullanarak veri alır, LlamaIndex kullanarak bilgi grafiği oluşturur ve sorgulama yaparak yanıt üretir. Bu mimari, gerçek zamanlı veri alma ve işleme için kullanılabilir.

---

## Building graphs from trees

## Ağaçlardan Grafik Oluşturma (Building Graphs from Trees)

Bir grafik (graph), kenarlar (edges veya arcs) ile birbirine bağlı düğümlerin (nodes veya vertices) bir koleksiyonudur. Düğümler varlıkları (entities), kenarlar ise bu varlıklar arasındaki ilişkileri veya bağlantıları temsil eder. Örneğin, bu bölümdeki kullanım senaryosunda düğümler çeşitli pazarlama stratejilerini (marketing strategies) temsil edebilir ve kenarlar bu stratejilerin nasıl birbirine bağlı olduğunu gösterebilir. Bu, yeni müşterilerin farklı pazarlama taktiklerinin genel iş hedeflerine ulaşmak için nasıl birlikte çalıştığını anlamalarına yardımcı olur, daha net iletişim ve daha etkili strateji planlaması sağlar.

## Kod Açıklaması

Aşağıdaki kod, Python'da NetworkX ve Matplotlib kullanarak bir ağaç yapısındaki ilişkileri görsel olarak temsil etmek için tasarlanmıştır. Belirli düğüm çiftlerinden bir yönlü grafik (directed graph) oluşturur, arkadaşlıkları kontrol eder ve bu ağacı özelleştirilmiş görsel özniteliklerle (visual attributes) görüntüler.

### Kod
```python
import networkx as nx
import matplotlib.pyplot as plt

# Düğüm çiftleri (pairs)
pairs = [('a', 'b'), ('b', 'e'), ('e', 'm'), ('m', 'p'), ('a', 'z'), ('b', 'q')]

# Arkadaşlık verileri (friends)
friends = {('a', 'b'), ('b', 'e'), ('e', 'm'), ('m', 'p')}

# build_tree_from_pairs fonksiyonu
def build_tree_from_pairs(pairs):
    G = nx.DiGraph()
    G.add_edges_from(pairs)
    root = None
    for node in G.nodes():
        if G.in_degree(node) == 0:
            root = node
            break
    return G, root

# check_relationships fonksiyonu
def check_relationships(pairs, friends):
    for pair in pairs:
        if pair in friends:
            print(f"Pair {pair}: friend")
        else:
            print(f"Pair {pair}: not friend")

# draw_tree fonksiyonu
def draw_tree(G, layout_choice, root, friends):
    pos = None
    if layout_choice == 'spring':
        pos = nx.spring_layout(G)
    elif layout_choice == 'circular':
        pos = nx.circular_layout(G)
    # Diğer layout seçenekleri buraya eklenebilir
    
    edge_colors = ['black' if edge in friends else 'red' for edge in G.edges()]
    edge_styles = ['solid' if edge in friends else 'dashed' for edge in G.edges()]
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color=edge_colors, style=edge_styles)
    plt.show()

# Ağacı oluştur
tree, root = build_tree_from_pairs(pairs)

# İlişkileri kontrol et
check_relationships(pairs, friends)

# Ağacı çiz
layout_choice = 'spring'
draw_tree(tree, layout_choice=layout_choice, root=root, friends=friends)
```

### Önemli Noktalar

*   `build_tree_from_pairs` fonksiyonu, düğüm çiftlerinden bir yönlü grafik oluşturur ve kök düğümü (root node) belirler.
*   `check_relationships` fonksiyonu, düğüm çiftleri arasındaki arkadaşlıkları kontrol eder ve sonuçları yazdırır.
*   `draw_tree` fonksiyonu, grafiği Matplotlib kullanarak çizer ve kenar stillerini (edge styles) arkadaşlık durumuna göre ayarlar.
*   `spring` layout seçeneği, düğümleri birbirine bağlı kenarların çekimi ve düğümlerin birbirini itmesiyle oluşan bir düzen içinde yerleştirir.

### Kullanım

1.  Düğüm çiftleri (`pairs`) ve arkadaşlık verileri (`friends`) tanımlanır.
2.  `build_tree_from_pairs` fonksiyonu kullanılarak ağaç oluşturulur.
3.  `check_relationships` fonksiyonu kullanılarak düğüm çiftleri arasındaki arkadaşlıklar kontrol edilir.
4.  `draw_tree` fonksiyonu kullanılarak ağaç çizilir.

Bu kod, ağaç yapısındaki ilişkileri görselleştirmek ve arkadaşlıkları temsil etmek için kullanılabilir. Farklı layout seçenekleri ve görsel öznitelikler kullanılarak daha karmaşık grafikler oluşturulabilir.

---

## Pipeline 1: Collecting and preparing the documents

## Pipeline 1: Dokümanları Toplama ve Hazırlama

Bu bölümdeki kod, Wikipedia'dan ihtiyacımız olan meta verileri alır, dokümanları alır, temizler ve Deep Lake vektör deposuna (vector store) eklenmeye hazır hale getirir. Bu süreç aşağıdaki şekilde gösterilmiştir:

### Pipeline 1 Akış Şeması
## Şekil 7.4: Pipeline 1 akış şeması

### Pipeline 1
Pipeline 1 iki not defteri içerir:
- `Wikipedia_API.ipynb`: Bu not defterinde, seçtiğimiz konu ile ilgili sayfaların URL'lerini, her sayfa için alıntıları (citations) da içerecek şekilde almak için Wikipedia API'sini uygulayacağız. Konumuz "pazarlama (marketing)" olarak belirlenmiştir.
- `Knowledge_Graph_Deep_Lake_LlamaIndex_OpenAI_RAG.ipynb`: Bu not defterinde, üç pipeline'ı da uygulayacağız. Pipeline 1'de, `Wikipedia_API` not defteri tarafından sağlanan URL'leri alacak, temizleyecek, yükleyecek ve birleştireceğiz.

İlk olarak, Wikipedia API'sini uygulayarak başlayacağız.

## Wikipedia API Uygulaması

Wikipedia API'sini kullanmak için aşağıdaki kod bloğunu kullanacağız:
```python
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError

def get_wikipedia_page(title):
    try:
        page = wikipedia.page(title)
        return page
    except DisambiguationError as e:
        print(f"DisambiguationError: {e}")
        return None
    except PageError as e:
        print(f"PageError: {e}")
        return None

# Konu ile ilgili sayfaları almak için Wikipedia API'sini kullanma
topic = "marketing"
page = get_wikipedia_page(topic)
```
Bu kodda, `wikipedia` kütüphanesini içe aktarıyoruz (`import wikipedia`) ve `DisambiguationError` ve `PageError` istisnalarını (`exceptions`) ele almak için `wikipedia.exceptions` modülünü içe aktarıyoruz.

`get_wikipedia_page` fonksiyonu, belirtilen başlık (`title`) ile ilgili Wikipedia sayfasını almak için kullanılır. Bu fonksiyon, `DisambiguationError` ve `PageError` istisnalarını yakalar ve hata mesajları yazdırır.

`topic` değişkenine "marketing" değerini atıyoruz ve `get_wikipedia_page` fonksiyonunu çağırarak ilgili sayfayı alıyoruz.

## Alınan Sayfaların URL'lerini Alma

Alınan sayfanın URL'lerini almak için aşağıdaki kod bloğunu kullanacağız:
```python
def get_urls(page):
    urls = []
    for link in page.links:
        try:
            linked_page = wikipedia.page(link)
            urls.append(linked_page.url)
        except DisambiguationError as e:
            print(f"DisambiguationError: {e}")
        except PageError as e:
            print(f"PageError: {e}")
    return urls

urls = get_urls(page)
```
Bu kodda, `get_urls` fonksiyonu, alınan sayfanın bağlantılarını (`links`) alır ve her bağlantı için ilgili sayfayı almak üzere `wikipedia.page` fonksiyonunu çağırır. Alınan sayfanın URL'sini (`url`) `urls` listesine ekler.

## Dokümanları Temizleme ve Birleştirme

Dokümanları temizlemek ve birleştirmek için aşağıdaki kod bloğunu kullanacağız:
```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    sentences = sent_tokenize(text)
    cleaned_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalpha()]
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]
        cleaned_sentence = ' '.join(words)
        cleaned_sentences.append(cleaned_sentence)
    return ' '.join(cleaned_sentences)

# Dokümanları temizleme ve birleştirme
cleaned_texts = []
for url in urls:
    try:
        page = wikipedia.page(url)
        text = page.content
        cleaned_text = clean_text(text)
        cleaned_texts.append(cleaned_text)
    except DisambiguationError as e:
        print(f"DisambiguationError: {e}")
    except PageError as e:
        print(f"PageError: {e}")

combined_text = ' '.join(cleaned_texts)
```
Bu kodda, `nltk` kütüphanesini içe aktarıyoruz (`import nltk`) ve `stopwords`, `word_tokenize`, `sent_tokenize` ve `WordNetLemmatizer` fonksiyonlarını kullanıyoruz.

`clean_text` fonksiyonu, metni (`text`) temizler. Öncelikle, cümleleri (`sentences`) ayırır ve her cümleyi kelimelere (`words`) böler. Kelimeleri küçük harfe (`lower`) çevirir, alfabe olmayan karakterleri (`isalpha`) temizler, stop words'leri (`stop_words`) temizler ve kelimeleri (`lemmatize`) birleştirir.

Dokümanları temizledikten sonra, temizlenmiş metinleri (`cleaned_texts`) birleştirerek (`combined_text`) elde ediyoruz.

## Önemli Noktalar
- Wikipedia API'sini kullanarak konu ile ilgili sayfaları almak.
- Alınan sayfaların URL'lerini almak.
- Dokümanları temizlemek ve birleştirmek için `nltk` kütüphanesini kullanmak.
- Temizlenmiş metinleri birleştirmek.

## Kullanılan Kodlar
- `wikipedia` kütüphanesini içe aktarmak: `import wikipedia`
- `nltk` kütüphanesini içe aktarmak: `import nltk`
- `DisambiguationError` ve `PageError` istisnalarını ele almak: `from wikipedia.exceptions import DisambiguationError, PageError`
- `stopwords`, `word_tokenize`, `sent_tokenize` ve `WordNetLemmatizer` fonksiyonlarını kullanmak: `from nltk.corpus import stopwords`, `from nltk.tokenize import word_tokenize, sent_tokenize`, `from nltk.stem import WordNetLemmatizer`

---

## Retrieving Wikipedia data and metadata

## Vikipedi Verilerini ve Metaverilerini Alma (Retrieving Wikipedia Data and Metadata)

Bu bölümde, belirli bir konu hakkında bilgi almak, alınan metni tokenize etmek (tokenization) ve Vikipedi makalelerinden alıntıları yönetmek için Vikipedi API'si ile etkileşim kuran bir program oluşturacağız.

### Gerekli Kütüphanelerin Kurulumu

İlk adım, gerekli `wikipediaapi` kütüphanesini kurmaktır:
```python
try:
    import wikipediaapi
except:
    !pip install Wikipedia-API==0.6.0
    import wikipediaapi
```
Bu kod, `wikipediaapi` kütüphanesini kurmayı dener. Eğer kütüphane yüklü değilse, `!pip install` komutu ile kurulur.

### Tokenizasyon Fonksiyonu

Tokenizasyon fonksiyonu, bir metnin içerdiği token sayısını saymak için kullanılır:
```python
import nltk
from nltk.tokenize import word_tokenize

def nb_tokens(text):
    # Daha gelişmiş tokenizasyon, noktalama işaretlerini içerir
    tokens = word_tokenize(text)
    return len(tokens)
```
Bu fonksiyon, bir metni girdi olarak alır ve NLTK kütüphanesini kullanarak metnin içerdiği token sayısını döndürür.

### Vikipedi API'sini Kurma

Vikipedi API'sini kullanmak için bir örnek oluşturmalıyız:
```python
wiki = wikipediaapi.Wikipedia(
    language='en',  # İngilizce için 'en'
    user_agent='Knowledge/1.0 ([USER AGENT EMAIL)'  # Kullanıcı aracısı bilgisi
)
```
Bu örnek, İngilizce Vikipedi için oluşturulmuştur ve bir kullanıcı aracısı bilgisi gerektirir.

### Ana Konu ve Dosya Adını Tanımlama

Ana konu ve dosya adını tanımlayalım:
```python
topic = "Marketing"  # Konu
filename = "Marketing"  # Dosya adı
maxl = 100  # Maksimum bağlantı sayısı
```
Bu parametreler, veri alma işleminin konusunu, dosya adını ve maksimum bağlantı sayısını tanımlar.

### Vikipedi Sayfasının Özetini Alma

Vikipedi sayfasının özetini alalım:
```python
page = wiki.page(topic)
if page.exists() == True:
    print("Page - Exists: %s" % page.exists())
    summary = page.summary
    nbt = nb_tokens(summary)
    print("Number of tokens: ", nbt)
    wrapped_text = textwrap.fill(summary, width=60)
    print(wrapped_text)
else:
    print("Page does not exist")
```
Bu kod, Vikipedi sayfasının var olup olmadığını kontrol eder, özetini alır ve token sayısını hesaplar.

### Bağlantıları Alma

Vikipedi sayfasındaki bağlantıları alalım:
```python
links = page.links
urls = []
counter = 0
for link in links:
    try:
        counter += 1
        print(f"Link {counter}: {link}")
        summary = wiki.page(link).summary
        print(f"Link: {link}")
        print(wiki.page(link).fullurl)
        urls.append(wiki.page(link).fullurl)
        print(f"Summary: {summary}")
        if counter >= maxl:
            break
    except:
        # Var olmayan sayfaları yoksay
        pass
print(counter)
print(urls)
```
Bu kod, Vikipedi sayfasındaki bağlantıları alır, özetlerini ve URL'lerini yazdırır.

### Alıntı Dosyasını Oluşturma

Alıntı dosyasını oluşturalım:
```python
from datetime import datetime

links = page.links
fname = filename + "_citations.txt"
with open(fname, "w") as file:
    file.write(f"Citation. In Wikipedia, The Free Encyclopedia. Pages retrieved from the following Wikipedia contributors on {datetime.now()}\n")
    file.write("Root page: " + page.fullurl + "\n")
    counter = 0
    urls = []
    # ...
```
Bu kod, alıntı dosyasını oluşturur ve içine gerekli bilgileri yazar.

### URL Dosyasını Oluşturma

URL dosyasını oluşturalım:
```python
ufname = filename + "_urls.txt"
with open(ufname, 'w') as file:
    for url in urls:
        file.write(url + '\n')
print("URLs have been written to urls.txt")
```
Bu kod, URL dosyasını oluşturur ve içine URL'leri yazar.

Bu adımların sonunda, Vikipedi verilerini ve metaverilerini almış olduk.

---

## Preparing the data for upsertion

## Veri Ekleme (Upsertion) için Verilerin Hazırlanması
Wikipedia_API.ipynb notebook'unda Wikipedia API tarafından sağlanan URL'ler, bu bölümde Knowledge_Graph_ Deep_Lake_LlamaIndex_OpenAI_RAG.ipynb notebook'unda işlenecektir. Bu notebook'un Installing the environment (Çevreyi Kurma) bölümü, Chapter 2 (Bölüm 2) ve Chapter 3 (Bölüm 3) 'teki ilgili bölümlerle neredeyse aynıdır.

## Parametrelerin Tanımlanması
Bu bölümde, workflow'un stratejisini tanımlamak için notebook'un Scenario (Senaryo) bölümüne gidilir. Aşağıdaki parametreler tanımlanır:
```python
# Dosya yönetimi için dosya adı
graph_name = "Marketing"

# Vektör deposu ve veri kümesi için yol
db = "hub://denis76/marketing01"
vector_store_path = db
dataset_path = db

# Veri ekleme (upsertion) için parametreler
pop_vs = True  # True ise veri ekler, False ise veri eklemeyi atlar
ow = True  # True ise mevcut veri kümesini üzerine yazar, False ise ekler
```
Bu parametreler, notebook'taki üç pipeline'ın davranışını belirler.

*   `graph_name="Marketing"`: Okunacak ve yazılacak dosyaların prefix'i (konusu).
*   `db="hub://denis76/marketing01"`: Deep Lake vektör deposunun adı. İstediğiniz veri kümesi adını seçebilirsiniz.
*   `vector_store_path = db`: Vektör deposunun yolu.
*   `dataset_path = db`: Vektör deposunun veri kümesinin yolu.
*   `pop_vs=True`: True ise veri eklemeyi (upsertion) etkinleştirir, False ise devre dışı bırakır.
*   `ow=True`: True ise mevcut veri kümesini üzerine yazar, False ise ekler.

## Pipeline 1: Dokümanları Toplama ve Hazırlama
Ardından, Pipeline 1: Collecting and preparing the documents (Dokümanları Toplama ve Hazırlama) bölümünü çalıştırabiliriz. Program, önceki bölümde oluşturulan URL listesini indirecektir:
```python
if pop_vs == True:
    directory = "Chapter07/citations"
    file_name = graph_name + "_urls.txt"
    download(directory, file_name)
```
Bu kod, `pop_vs` parametresi True ise, belirtilen dizindeki `_urls.txt` dosyasını indirecektir.

Daha sonra, dosya okunur ve URL'ler `urls` adlı bir listede depolanır. Pipeline 1: Collecting and preparing the documents bölümünün geri kalan kodu, Chapter 3 (Bölüm 3) 'teki Deep_Lake_LlamaIndex_OpenAI_RAG.ipynb notebook'undaki işlemlerin aynısını takip eder.

Chapter 3 (Bölüm 3) 'te, web sayfalarının URL'leri manuel olarak bir listeye giriliyordu. Kod, URL listesindeki içeriği getirecektir. Program daha sonra Deep Lake vektör deposunu doldurmak için verileri temizler ve hazırlar.

## Kod Açıklaması
`download(directory, file_name)` fonksiyonu, belirtilen dizindeki dosyayı indirir. Bu fonksiyonun tanımı notebook'ta bulunmalıdır.

`pop_vs` ve `ow` parametreleri, veri ekleme (upsertion) işleminin nasıl gerçekleştirileceğini belirler. `pop_vs=True` ise veri eklemeyi etkinleştirir, `ow=True` ise mevcut veri kümesini üzerine yazar.

Veri ekleme (upsertion) işlemi için gerekli olan import ifadeleri aşağıdaki gibidir:
```python
import os
import requests
from pathlib import Path
```
Bu import ifadeleri, dosya indirme ve URL'lerle çalışmak için gerekli olan kütüphaneleri içe aktarır.

Tüm kod aşağıdaki gibidir:
```python
# Dosya yönetimi için dosya adı
graph_name = "Marketing"

# Vektör deposu ve veri kümesi için yol
db = "hub://denis76/marketing01"
vector_store_path = db
dataset_path = db

# Veri ekleme (upsertion) için parametreler
pop_vs = True  # True ise veri ekler, False ise veri eklemeyi atlar
ow = True  # True ise mevcut veri kümesini üzerine yazar, False ise ekler

if pop_vs == True:
    directory = "Chapter07/citations"
    file_name = graph_name + "_urls.txt"
    download(directory, file_name)
```

---

## Pipeline 2: Creating and populating the Deep Lake vector store

## Pipeline 2: Derin Göl (Deep Lake) Vektör Deposu Oluşturma ve Doldurma
Bu bölümdeki Deep_Lake_LlamaIndex_OpenAI_RAG.ipynb not defterinde (notebook) oluşturulan pipeline, 3. Bölüm'deki Pipeline 2 koduyla oluşturulmuştur. Pipeline'ları bileşenler olarak oluşturarak, bunları diğer uygulamalara hızla yeniden amaçlandırabileceğimizi ve uyarlayabileceğimizi görebiliriz. Ayrıca, Activeloop Deep Lake, varsayılan parçalama (chunking), embedding ve upserting fonksiyonlarına sahiptir, bu da çeşitli türlerde yapılandırılmamış verileri (örneğin, Wikipedia belgeleri) sorunsuz bir şekilde entegre etmeyi sağlar.

### Önemli Noktalar:
* Pipeline'lar bileşen olarak oluşturulur ve yeniden kullanılabilir.
* Activeloop Deep Lake, yapılandırılmamış verileri entegre etmek için kullanışlıdır.
* `display_record(record_number)` fonksiyonu, veri kayıtlarını görüntülemek için kullanılır.

## Kod Açıklaması
Aşağıdaki kod, Deep Lake vektör deposunu oluşturmak ve doldurmak için kullanılır:
```python
import deeplake
from deeplake.constants import MB
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import DeepLakeVectorStore

# Deep Lake vektör deposu oluşturma
dataset_path = "./data/deeplake"
vector_store = DeepLakeVectorStore(
    dataset_path=dataset_path,
    overwrite=True,
    verbose=False
)

# Verileri yükleme
documents = SimpleDirectoryReader("./data/wiki").load_data()

# Vektör deposunu doldurma
index = VectorStoreIndex.from_documents(
    documents=documents,
    vector_store=vector_store,
    show_progress=True
)
```
### Kodun Açıklaması:
1. `deeplake` ve `llama_index` kütüphaneleri import edilir.
2. Deep Lake vektör deposu oluşturulur ve `dataset_path` parametresi ile yolu belirlenir.
3. `SimpleDirectoryReader` sınıfı kullanılarak `./data/wiki` dizinindeki belgeler yüklenir.
4. `VectorStoreIndex.from_documents` metodu kullanılarak vektör deposu doldurulur.

## Kayıtları Görüntüleme
`display_record(record_number)` fonksiyonu kullanılarak kayıtlar görüntülenebilir:
```python
display_record(record_number=0)
```
Bu fonksiyon, ID, metadata, metin ve embedding gibi bilgileri görüntüler:
```
ID: ['a61734be-fe23-421e-9a8b-db6593c48e08']
Metadata:
file_path: /content/data/24-hour_news_cycle.txt
file_name: 24-hour_news_cycle.txt
file_type: text/plain
file_size: 2763
creation_date: 2024-07-05
last_modified_date: 2024-07-05
…
Text: ['24hour investigation and reporting of news concomitant with fastpaced lifestyles This article is about the fastpaced cycle of news media in technologically advanced societies.']
Embedding: [-0.00040736704249866307, 0.009565318934619427, 0.015906672924757004, -0.009085721336305141, …]
```
Bu şekilde, Pipeline 2 bileşenini başarıyla yeniden amaçlandırdık ve grafik bilgi dizinini oluşturmaya hazırız.

---

## Pipeline 3: Knowledge graph index-based RAG

## Pipeline 3: Bilgi Grafiği İndeksi Tabanlı RAG (Knowledge Graph Index-Based RAG)

Bu bölümde, Bilgi Grafiği İndeksi Tabanlı RAG (Knowledge Graph Index-Based RAG) pipeline'ı oluşturacağız ve onunla etkileşime gireceğiz. Aşağıdaki şekilde gösterildiği gibi, yapacak çok işimiz var:

## Adımlar (Steps)

Aşağıdaki adımları takip edeceğiz:
* Bilgi grafiği indeksini (Knowledge Graph Index) oluşturma
* Grafiği görüntüleme (Display the Graph)
* Kullanıcı istemini (User Prompt) tanımlama
* LlamaIndex'in yerleşik LLM (Large Language Model) modelinin hiperparametrelerini (Hyperparameters) tanımlama
* Benzerlik puanı (Similarity Score) paketlerini kurma
* Benzerlik puanı fonksiyonlarını tanımlama
* Benzerlik fonksiyonları arasında örnek bir benzerlik karşılaştırması yapma
* Bir LLM yanıtının çıktı vektörlerini yeniden sıralama (Re-rank the Output Vectors)
* Değerlendirme örnekleri çalıştırma ve metrikler ve insan geri bildirim puanları uygulama
* Metrik hesaplamalarını çalıştırma ve görüntüleme

### Bilgi Grafiği İndeksini Oluşturma (Generating the Knowledge Graph Index)

İlk adım, bilgi grafiği indeksini oluşturmaktır. Bunu yapmak için, aşağıdaki kodu kullanacağız:
```python
from llama_index import KnowledgeGraphIndex
from llama_index.graph_stores import SimpleGraphStore

# Grafiği oluşturma
graph_store = SimpleGraphStore()
index = KnowledgeGraphIndex(
    nodes=[
        {"id": "node1", "text": "Bu bir örnek metin."},
        {"id": "node2", "text": "Bu başka bir örnek metin."},
    ],
    graph_store=graph_store,
    max_triplets=10,
    include_embeddings=True,
)

# İndeksi oluşturma
index.build_index()
```
Bu kod, `KnowledgeGraphIndex` sınıfını kullanarak bir bilgi grafiği indeksi oluşturur. `nodes` parametresi, grafikteki düğümleri (node) tanımlar. `graph_store` parametresi, grafiği depolamak için kullanılan bir nesne (object)dir. `max_triplets` parametresi, grafikteki maksimum triplet sayısını tanımlar. `include_embeddings` parametresi, düğümlerin embedding'lerini (gömme) içermeyi sağlar.

### Grafiği Görüntüleme (Displaying the Graph)

Grafiği görüntülemek için, aşağıdaki kodu kullanacağız:
```python
import networkx as nx
import matplotlib.pyplot as plt

# Grafiği oluşturma
G = nx.DiGraph()
G.add_nodes_from([node["id"] for node in index.nodes])
G.add_edges_from([(edge["source"], edge["target"]) for edge in index.edges])

# Grafiği görüntüleme
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos, with_labels=True, node_color="lightblue")
plt.show()
```
Bu kod, `networkx` kütüphanesini kullanarak grafiği oluşturur ve `matplotlib` kütüphanesini kullanarak grafiği görüntüler.

### Kullanıcı İstemini Tanımlama (Defining the User Prompt)

Kullanıcı istemini tanımlamak için, aşağıdaki kodu kullanacağız:
```python
user_prompt = "Örnek bir metin nedir?"
```
Bu kod, bir kullanıcı istemini tanımlar.

### LlamaIndex'in Yerleşik LLM Modelinin Hiperparametrelerini Tanımlama (Defining the Hyperparameters of LlamaIndex's In-Built LLM Model)

LlamaIndex'in yerleşik LLM modelinin hiperparametrelerini tanımlamak için, aşağıdaki kodu kullanacağız:
```python
from llama_index.llms import LLM

# LLM modelini oluşturma
llm = LLM(
    model_name="llama-index",
    max_tokens=512,
    temperature=0.7,
)

# Hiperparametreleri tanımlama
llm.config["max_tokens"] = 1024
llm.config["temperature"] = 0.5
```
Bu kod, LlamaIndex'in yerleşik LLM modelini oluşturur ve hiperparametrelerini tanımlar.

### Benzerlik Puanı Paketlerini Kurma (Installing the Similarity Score Packages)

Benzerlik puanı paketlerini kurmak için, aşağıdaki kodu kullanacağız:
```bash
pip install -U sentence-transformers
pip install -U faiss-cpu
```
Bu kod, `sentence-transformers` ve `faiss-cpu` paketlerini kurar.

### Benzerlik Puanı Fonksiyonlarını Tanımlama (Defining the Similarity Score Functions)

Benzerlik puanı fonksiyonlarını tanımlamak için, aşağıdaki kodu kullanacağız:
```python
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

# Benzerlik puanı modelini oluşturma
model = SentenceTransformer("all-MiniLM-L6-v2")

# Benzerlik puanı fonksiyonunu tanımlama
def similarity_score(query, response):
    query_embedding = model.encode(query)
    response_embedding = model.encode(response)
    return util.cos_sim(query_embedding, response_embedding)
```
Bu kod, `sentence-transformers` kütüphanesini kullanarak bir benzerlik puanı modeli oluşturur ve bir benzerlik puanı fonksiyonu tanımlar.

### Örnek Bir Benzerlik Karşılaştırması Yapma (Running a Sample Similarity Comparison)

Örnek bir benzerlik karşılaştırması yapmak için, aşağıdaki kodu kullanacağız:
```python
query = "Örnek bir metin nedir?"
response1 = "Bu bir örnek metin."
response2 = "Bu başka bir örnek metin."

# Benzerlik puanını hesaplama
score1 = similarity_score(query, response1)
score2 = similarity_score(query, response2)

print("Score 1:", score1)
print("Score 2:", score2)
```
Bu kod, örnek bir benzerlik karşılaştırması yapar ve benzerlik puanlarını hesaplar.

### Bir LLM Yanıtının Çıktı Vektörlerini Yeniden Sıralama (Re-Ranking the Output Vectors)

Bir LLM yanıtının çıktı vektörlerini yeniden sıralamak için, aşağıdaki kodu kullanacağız:
```python
# LLM yanıtını oluşturma
response = llm.generate(user_prompt)

# Çıktı vektörlerini yeniden sıralama
reranked_response = sorted(response, key=lambda x: similarity_score(user_prompt, x), reverse=True)

print("Reranked Response:", reranked_response)
```
Bu kod, bir LLM yanıtının çıktı vektörlerini yeniden sıralar.

### Değerlendirme Örnekleri Çalıştırma ve Metrikler ve İnsan Geri Bildirim Puanları Uygulama (Running Evaluation Samples and Applying Metrics and Human Feedback Scores)

Değerlendirme örnekleri çalıştırmak ve metrikler ve insan geri bildirim puanları uygulamak için, aşağıdaki kodu kullanacağız:
```python
# Değerlendirme örnekleri oluşturma
evaluation_samples = [
    {"query": "Örnek bir metin nedir?", "response": "Bu bir örnek metin."},
    {"query": "Örnek bir metin nedir?", "response": "Bu başka bir örnek metin."},
]

# Metrikleri hesaplama
metrics = []
for sample in evaluation_samples:
    score = similarity_score(sample["query"], sample["response"])
    metrics.append(score)

# İnsan geri bildirim puanları uygulama
human_feedback_scores = [0.8, 0.9]

# Değerlendirme sonuçlarını hesaplama
evaluation_results = []
for i, sample in enumerate(evaluation_samples):
    evaluation_results.append({
        "query": sample["query"],
        "response": sample["response"],
        "score": metrics[i],
        "human_feedback_score": human_feedback_scores[i],
    })

print("Değerlendirme Sonuçları:", evaluation_results)
```
Bu kod, değerlendirme örnekleri oluşturur, metrikleri hesaplar, insan geri bildirim puanları uygular ve değerlendirme sonuçlarını hesaplar.

### Metrik Hesaplamalarını Çalıştırma ve Görüntüleme (Running Metric Calculations and Displaying Them)

Metrik hesaplamalarını çalıştırmak ve görüntülemek için, aşağıdaki kodu kullanacağız:
```python
# Metrik hesaplamalarını çalıştırma
metrics = []
for sample in evaluation_samples:
    score = similarity_score(sample["query"], sample["response"])
    metrics.append(score)

# Metrikleri görüntüleme
print("Metrikler:", metrics)
```
Bu kod, metrik hesaplamalarını çalıştırır ve metrikleri görüntüler.

---

## Generating the knowledge graph index

## Bilgi Grafiği İndeksini (Knowledge Graph Index) Oluşturma
Bir dizi belgeden (documents) bilgi grafiği indeksini oluşturmak için `llama_index.core` modülünden `KnowledgeGraphIndex` sınıfını kullanacağız. Ayrıca, performans değerlendirmesi için indeks oluşturma sürecini zamanlayacağız.

### Zamanlayıcı Başlatma
İndeks oluşturma işlemi oldukça uzun sürebileceği için zaman ölçümü önemlidir. 
```python
from llama_index.core import KnowledgeGraphIndex
import time

# Zamanlayıcıyı başlat
start_time = time.time()
```

### Bilgi Grafiği İndeksini Oluşturma
`KnowledgeGraphIndex` sınıfının `from_documents` metodunu kullanarak embeddings içeren bir bilgi grafiği indeksi oluşturacağız. 
```python
# Belge kümesinden bilgi grafiği indeksini oluştur
graph_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,  # Bellek kullanımı ve işleme süresini optimize etmek için chunk başına triplet sayısını sınırlar
    include_embeddings=True,  # Embeddings'i dahil et
)
```

### Zamanlayıcıyı Durdurma ve İndeks Oluşturma Süresini Hesaplama
Zamanlayıcıyı durdurup indeks oluşturma süresini hesaplayacağız.
```python
# Zamanlayıcıyı durdur
end_time = time.time()

# İndeks oluşturma süresini hesapla ve yazdır
elapsed_time = end_time - start_time
print(f"İndeks oluşturma süresi: {elapsed_time:.4f} saniye")
print(type(graph_index))
```

### Çıktı
İndeks oluşturma süresi ve graf tipi yazdırılır.
```
İndeks oluşturma süresi: 371.9844 saniye
<class 'llama_index.core.indices.knowledge_graph.base.KnowledgeGraphIndex'>
```

### Sorgu Motorunu (Query Engine) Yapılandırma
Bilgi grafiği indeksi için bir sorgu motoru yapılandıracağız ve benzerlik (similarity), yanıt sıcaklığı (response temperature) ve çıktı uzunluğu (output length) parametrelerini yöneteceğiz.
```python
# Benzerlik için üst k değeri
k = 3

# Sıcaklık parametresi
temp = 0.1

# Çıktı uzunluğu
mt = 1024

# Sorgu motorunu oluştur
graph_query_engine = graph_index.as_query_engine(
    similarity_top_k=k, 
    temperature=temp, 
    num_output=mt
)
```

### Parametrelerin Anlamı
- `k=3`: En üst benzerlik sonuçlarının sayısını belirler.
- `temp=0.1`: Sorgu motorunun yanıt oluşturmasındaki rastgeleliği kontrol eder. Düşük değerler daha kesin, yüksek değerler daha yaratıcı sonuçlar verir.
- `mt=1024`: Çıktı için maksimum token sayısını belirler ve oluşturulan yanıtların uzunluğunu tanımlar.

### Sonuç
Bilgi grafiği indeksi ve sorgu motoru hazır. Grafiği görüntüleyebiliriz. 

Önemli noktalar:
* `KnowledgeGraphIndex` sınıfını kullanarak belge kümesinden bilgi grafiği indeksi oluşturduk.
* İndeks oluşturma sürecini zamanladık.
* Sorgu motorunu yapılandırdık ve benzerlik, yanıt sıcaklığı ve çıktı uzunluğu parametrelerini yönettik.
* `from_documents` metodunu kullanarak embeddings içeren bir bilgi grafiği indeksi oluşturduk.
* `as_query_engine` metodunu kullanarak sorgu motorunu oluşturduk.

---

## Displaying the graph

## Grafiğin Gösterilmesi
Pyvis.network kütüphanesini kullanarak etkileşimli bir ağ görselleştirmesi (interactive network visualization) oluşturacağız. Bu kütüphane, Python'da etkileşimli ağ görselleştirmeleri oluşturmak için kullanılır.

### Grafiği Oluşturma
Öncelikle, bir grafik örneği (graph instance) `g` oluşturacağız.

```python
from pyvis.network import Network
g = graph_index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
```
Bu kodda:
- `graph_index.get_networkx_graph()` fonksiyonu, bir NetworkX grafiği döndürür.
- `Network` sınıfı, pyvis.network kütüphanesinden içe aktarılır (import).
- `net = Network(notebook=True, cdn_resources="in_line", directed=True)` satırı, bir Network örneği oluşturur. 
  - `notebook=True` parametresi, grafiğin Jupyter Notebook içinde görüntülenmesini sağlar.
  - `cdn_resources="in_line"` parametresi, gerekli kaynakların (resources) satır içi (in-line) yüklenmesini sağlar.
  - `directed=True` parametresi, grafiğin yönlü (directed) olduğunu belirtir.
- `net.from_nx(g)` fonksiyonu, NetworkX grafiğini pyvis.network grafiğine dönüştürür.

### Düğüm ve Kenar Özelliklerini Ayarlama
Düğüm ve kenar özelliklerini (node and edge properties) ayarlayacağız.

```python
for node in net.nodes:
    node['color'] = 'lightgray'
    node['size'] = 10

for edge in net.edges:
    edge['color'] = 'black'
    edge['width'] = 1
```
Bu kodda:
- `net.nodes` ve `net.edges` sırasıyla grafiğin düğümlerini (nodes) ve kenarlarını (edges) temsil eder.
- Düğümlerin rengini (color) 'lightgray' ve boyutunu (size) 10 olarak ayarladık.
- Kenarların rengini (color) 'black' ve genişliğini (width) 1 olarak ayarladık.

### Grafiği HTML Dosyasına Kaydetme
Yönlü grafiği (directed graph) bir HTML dosyasına kaydedeceğiz.

```python
fgraph = "Knowledge_graph_" + graph_name + ".html"
net.write_html(fgraph)
print(fgraph)
```
Bu kodda:
- `graph_name` değişkeni, senaryo bölümünde tanımlanmıştır.
- `net.write_html(fgraph)` fonksiyonu, grafiği bir HTML dosyasına yazar.

### Grafiği Notebook İçinde Görüntüleme
Grafiği notebook içinde bir HTML dosyası olarak görüntüleyeceğiz.

```python
from IPython.display import HTML

with open(fgraph, 'r') as file:
    html_content = file.read()

display(HTML(html_content))
```
Bu kodda:
- `IPython.display` modülünden `HTML` sınıfı içe aktarılır.
- `open` fonksiyonu, HTML dosyasını okur (read mode).
- `display` fonksiyonu, HTML içeriğini notebook içinde görüntüler.

Artık grafiği indirebilir ve tarayıcınızda görüntüleyerek etkileşimde bulunabilirsiniz. Ayrıca, aşağıdaki şekilde gösterildiği gibi notebook içinde de görselleştirebilirsiniz.

## Şekil 7.6: Bilgi Grafiği (Knowledge Graph)

Bilgi grafiği ile etkileşimde bulunmaya hazırız.

---

## Interacting with the knowledge graph index

## Bilgi Grafiği İndeksiyle Etkileşim (Interacting with the Knowledge Graph Index)

Bu bölümde, daha önce Pipeline 3: İndeks Tabanlı RAG (Index-based RAG) bölümünde yaptığımız gibi, sorguyu yürütmek için gerekli işlevselliği tanımlayacağız.

### Sorgu Yürütme İşlevi (execute_query Function)

Sorguyu yürütecek olan `execute_query` işlevini tanımlayacağız. Bu işlev, daha önce oluşturduğumuz `graph_query_engine.query(user_input)` kodunu kullanarak sorguyu yürütür ve yanıtı döndürür.

```python
response = graph_query_engine.query(user_input)
```

Bu kod, `graph_query_engine` nesnesinin `query` metodunu çağırarak `user_input` değişkeninde saklanan sorguyu yürütür.

### Sorgu Yürütme ve Zaman Ölçümü (Query Execution and Time Measurement)

Sorguyu yürütürken aynı zamanda yürütme süresini de ölçüyoruz.

```python
user_query = "What is the primary goal of marketing for the consumer market?"
response = execute_query(user_query)
```

Bu kod, `user_query` değişkeninde saklanan sorguyu `execute_query` işlevine geçirerek yürütür ve yanıtı `response` değişkeninde saklar.

### Çıktı (Output)

Çıktı, Wikipedia verileriyle oluşturduğumuz en iyi vektörleri ve yürütme süresini içerir.

```
Query execution time: 2.4789 seconds
The primary goal of marketing for the consumer market is to effectively target consumers, understand their behavior, preferences, and needs, and ultimately influence their purchasing decisions.
```

### Benzerlik Skor Paketlerini Yükleme ve Benzerlik Hesaplama İşlevlerini Tanımlama

Şimdi, benzerlik skor paketlerini yükleyeceğiz ve gerekli benzerlik hesaplama işlevlerini tanımlayacağız.

Önemli noktalar:

* `execute_query` işlevi, sorguyu yürütmek için kullanılır.
* `graph_query_engine.query(user_input)` kodu, sorguyu yürütür.
* Yürütme süresi ölçülür.
* Benzerlik skor paketleri yüklenecek ve benzerlik hesaplama işlevleri tanımlanacaktır.

Gerekli import işlemleri:

```python
# gerekli import işlemleri burada yapılmalıdır
```

Kodların kullanımı:

* `execute_query` işlevi, sorguyu yürütmek için kullanılır.
* `graph_query_engine.query(user_input)` kodu, sorguyu yürütür.
* `user_query` değişkeni, sorguyu saklamak için kullanılır.
* `response` değişkeni, yanıtı saklamak için kullanılır.

Ek bilgiler:

* `graph_query_engine` nesnesi, sorguyu yürütmek için kullanılan bir nesnedir.
* `query` metodu, sorguyu yürütmek için kullanılan bir metoddur.
* Benzerlik skor paketleri, benzerlik hesaplama işlevlerini tanımlamak için kullanılır.

---

## Installing the similarity score packages and defining the functions

## Benzerlik Skoru Paketlerini Kurma ve Fonksiyonları Tanımlama (Installing the Similarity Score Packages and Defining the Functions)

Bu bölümde, benzerlik skoru paketlerini kurma ve fonksiyonları tanımlama adımlarını anlatacağız. İlk olarak, Google Colab'da Secrets sekmesinde saklanan Hugging Face token'ını alacağız.

### Hugging Face Token'ını Alma

```python
from google.colab import userdata
userdata.get('HF_TOKEN')
```

Bu kod, Google Colab'da Secrets sekmesinde saklanan Hugging Face token'ını alır. Ağustos 2024 itibariyle, bu token sentence-transformers için isteğe bağlıdır, bu nedenle bu kodu yorum satırına alabilirsiniz.

### Sentence-Transformers Paketini Kurma

```bash
!pip install sentence-transformers==3.0.1
```

Bu komut, sentence-transformers paketini kurar. Bu paket, metinler arasındaki benzerliği hesaplamak için kullanılır.

### Kosinüs Benzerliği Fonksiyonunu Tanımlama

```python
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity_with_embeddings(text1, text2):
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    similarity = cosine_similarity([embeddings1], [embeddings2])
    return similarity[0][0]
```

Bu kod, kosinüs benzerliği fonksiyonunu tanımlar. Bu fonksiyon, iki metin arasındaki benzerliği hesaplamak için kullanılır.

*   `model = SentenceTransformer('all-MiniLM-L6-v2')`: Bu satır, SentenceTransformer modelini 'all-MiniLM-L6-v2' modeli ile başlatır.
*   `embeddings1 = model.encode(text1)` ve `embeddings2 = model.encode(text2)`: Bu satırlar, girdi metinlerini vektör temsillerine (embeddings) dönüştürür.
*   `similarity = cosine_similarity([embeddings1], [embeddings2])`: Bu satır, iki vektör arasındaki kosinüs benzerliğini hesaplar.
*   `return similarity[0][0]`: Bu satır, benzerlik skorunu döndürür.

### Gerekli Kütüphaneleri İçe Aktarma

```python
import time
import textwrap
import sys
import io
```

Bu kod, gerekli kütüphaneleri içe aktarır. Bu kütüphaneler, çeşitli işlemler için kullanılır.

Artık bir benzerlik fonksiyonumuz var ve bunu yeniden sıralama (re-ranking) için kullanabiliriz.

---

## Re-ranking

## Yeniden Sıralama (Re-ranking)
Yeniden sıralama, bir sorgunun yanıtını en üstteki sonuçlarını yeniden sıralayarak muhtemelen daha iyi olan diğerlerini seçme işlemidir.

## Kullanılan Kodlar ve Açıklamalar
```python
user_query = "Which experts are often associated with marketing theory?"
```
Bu kod, yapılan sorguyu temsil eder.

```python
start_time = time.time()
```
Bu kod, sorgunun yürütülme zamanını kaydetmeye başlar.

```python
response = execute_query(user_query)
```
Bu kod, sorguyu yürütür.

```python
end_time = time.time()
```
Bu kod, zamanlayıcıyı durdurur ve sorgunun yürütülme zamanı görüntülenir.

```python
for idx, node_with_score in enumerate(response.source_nodes):
```
Bu kod, yanıttaki tüm düğümleri almak için yanıta göre yinelenir.

```python
similarity_score3 = calculate_cosine_similarity_with_embeddings(text1, text2)
```
Bu kod, kullanıcı sorgusu ile yanıttan alınan düğümlerdeki metin arasındaki benzerlik puanını hesaplar.

```python
best_score = similarity_score3
```
Bu kod, bulunan en iyi benzerlik puanını saklar.

```python
print(textwrap.fill(str(best_text), 100))
```
Bu kod, en iyi yeniden sıralanmış sonucu görüntüler.

## Önemli Noktalar
* Yeniden sıralama, sorgunun yanıtını iyileştirmek için kullanılır.
* Benzerlik puanı, kullanıcı sorgusu ile yanıttaki metin arasındaki benzerliği ölçer.
* Yeniden sıralama, farklı metrikler kullanılarak yapılabilir.
* İnsan geri bildirimi puanları, matematiksel metriklerin yetersiz kaldığı durumlarda kullanılabilir.

## Yeniden Sıralama Örneği
İlk sorgu yanıtı:
"Psychologists, cultural anthropologists, and market researchers are often associated with marketing theory."
Yeniden sıralanmış yanıt:
"Best Rank: 2
Best Score: 0.5217772722244263
[…In 1380 the German textile manufacturer Johann Fugger Johann Fugger Daniel Defoe Daniel Defoe travelled from Augsburg to Graben in order to gather information on the international textile industry… During this period   a London merchant published information on trade and economic resources of England and Scotland…]"
Yeniden sıralanmış yanıt, daha uzun ve ham belge içeriği içerir.

## İyileştirme Yöntemleri
* Farklı promptlar kullanmak
* Doküman eklemek veya silmek
* İnsan geri bildirimi puanları kullanmak
* LLM'i ince ayar yapmak (Fine-Tuning)

## Kullanılan Kütüphaneler ve Import Kısımları
```python
import time
import textwrap
```
Bu kod, kullanılan kütüphaneleri import eder.

## Notlar
Yeniden sıralama, sorgunun yanıtını iyileştirmek için kullanılan bir tekniktir. Farklı metrikler ve yöntemler kullanılarak yapılabilir. İnsan geri bildirimi puanları, matematiksel metriklerin yetersiz kaldığı durumlarda kullanılabilir.

---

## Example metrics

## Örnek Metrikler (Example Metrics)
Bilgi grafiği indeksinin sorgu motorunu değerlendirmek için, on örnek çalıştıracağız ve puanları takip edeceğiz.

## Puan Takibi (Tracking Scores)
İnsan geri bildirim puanlarını takip etmek için `rscores` (`rscores`) listesi kullanılırken, benzerlik fonksiyonu puanlarını takip etmek için `scores=[]` (`scores=[]`) listesi kullanılır.

```python
# Boş bir dizi oluştur
rscores = []  # İnsan geri bildirim puanları için boş bir skor
scores = []   # Benzerlik fonksiyonu puanları için boş bir skor
```

## Örnek Yapısı (Example Structure)
Her bir örnek aynı yapıya sahiptir:
- `user_query` (`user_query`): Sorgu motoruna giriş metni
- `elapsed_time` (`elapsed_time`): Sistemin yanıtının zaman ölçümünün sonucu
- `response = execute_query(user_query)` (`response = execute_query(user_query)`): Sorguyu çalıştırır

## Benzerlik Fonksiyonu ve İnsan Geri Bildirimi (Similarity Function and Human Feedback)
Bu örnekte, benzerlik fonksiyonu çalıştırılır ve insanlardan bir puan alınır.

```python
text1 = str(response)  # Sorgu motorunun cevabı
text2 = user_query      # Kullanıcı sorgusu
similarity_score3 = calculate_cosine_similarity_with_embeddings(text1, text2)
print(f"Cosine Similarity Score with sentence transformer: {similarity_score3:.3f}")
scores.append(similarity_score3)  # Benzerlik puanını scores listesine ekler
human_feedback = 0.75  # İnsan benzerlik değerlendirmesi
rscores.append(human_feedback)  # İnsan puanını rscores listesine ekler
```

## Kod Açıklaması (Code Explanation)
- `text1`: Sorgu motorunun cevabı
- `text2`: Kullanıcı sorgusu
- `similarity_score3`: Kosinüs benzerlik puanı
- `scores.append(similarity_score3)`: Benzerlik puanını `scores` listesine ekler
- `human_feedback`: İnsan benzerlik değerlendirmesi
- `rscores.append(human_feedback)`: İnsan puanını `rscores` listesine ekler

## Örnek Çıktıları (Example Outputs)
LLM'ler (Large Language Models) stokastik algoritmalardır. Bu nedenle, yanıtlar ve puanlar bir çalıştırma ile diğer arasında değişebilir.

### Örnek 1 (Example 1)
- Kullanıcı Sorgusu: Pazarlama teorisi ile hangi uzmanlar sıklıkla ilişkilendirilir?
- Cevap: Psikologlar, kültürel antropologlar ve diğer davranış bilimleri uzmanları sıklıkla pazarlama teorisi ile ilişkilendirilir.
- Kosinüs Benzerlik Puanı: 0.809
- İnsan Geri Bildirimi: 0.75

### Örnek 3 (Example 3)
- Kullanıcı Sorgusu: B2B ve B2C arasındaki fark nedir?
- Cevap: B2B işletmeleri ürün ve hizmetleri diğer şirketlere satarken, B2C işletmeleri doğrudan müşterilere satar.
- Kosinüs Benzerlik Puanı: 0.760
- İnsan Geri Bildirimi: 0.8

### Örnek 7 (Example 7)
- Kullanıcı Sorgusu: Tarım Pazarlama Servisi (AMS) hangi emtia programlarını sürdürür?
- Cevap: Tarım Pazarlama Servisi (AMS) beş emtia alanında programlar sürdürür: pamuk ve tütün, süt, meyve ve sebze, canlı hayvan ve tohum, ve kümes hayvanları.
- Kosinüs Benzerlik Puanı: 0.904
- İnsan Geri Bildirimi: 0.9

## Metrik Hesaplamaları (Metric Calculations)
Kosinüs benzerlik puanları ve insan geri bildirim puanları üzerinde metrik hesaplamaları yapacağız.

Gerekli import işlemleri:
```python
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
```

`calculate_cosine_similarity_with_embeddings` fonksiyonu için gerekli kod:
```python
def calculate_cosine_similarity_with_embeddings(text1, text2):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode([text1, text2])
    cosine_score = util.cos_sim(embeddings[0:1], embeddings[1:2])
    return cosine_score.item()
```

---

## Metric calculation and display

## Metrik Hesaplaması ve Gösterimi
Örneklerin kosinüs benzerlik skorları (Cosine Similarity Scores) `scores` değişkeninde saklanır:
```python
print(len(scores), scores)
```
Bu kod, `scores` listesinin uzunluğunu ve içeriğini yazdırır. Örneğin:
```
10 [0.808918, 0.720165, 0.7599532, 0.8513956, 0.5457667, 0.6963912, 0.9036964, 0.44829217, 0.59976315, 0.47448665]
```
Aynı örnekler için insan geri bildirimi skorları (Human Feedback Scores) `rscores` değişkeninde saklanır:
```python
print(len(rscores), rscores)
```
Bu kod, `rscores` listesinin uzunluğunu ve içeriğini yazdırır. Örneğin:
```
10 [0.75, 0.5, 0.8, 0.9, 0.65, 0.8, 0.9, 0.2, 0.2, 0.9]
```
Cevapları değerlendirmek için metrikler uygulanır:
```python
import numpy as np

mean_score = np.mean(scores)
median_score = np.median(scores)
std_deviation = np.std(scores)
variance = np.var(scores)
min_score = np.min(scores)
max_score = np.max(scores)
range_score = max_score - min_score
percentile_25 = np.percentile(scores, 25)
percentile_75 = np.percentile(scores, 75)
iqr = percentile_75 - percentile_25
```
Bu kod, `scores` listesi için çeşitli metrikler hesaplar:

*   **Ortalama (Mean)**: Verilerin toplamının veri sayısına bölünmesiyle elde edilir. Veri setinin merkezi değerini verir.
*   **Medyan (Median)**: Veriler küçükten büyüğe sıralandığında ortadaki değerdir. Veri setinin merkezi noktasını verir ve uç değerlerden (outliers) etkilenmez.
*   **Standart Sapma (Standard Deviation)**: Her bir veri noktasının ortalamadan ne kadar uzaklaştığını ölçer. Veri setinin ne kadar yayıldığını gösterir.
*   **Varyans (Variance)**: Standart sapmanın karesidir. Veri setinin yayılımını ölçer.
*   **Minimum (Minimum)**: Veri setindeki en küçük değerdir.
*   **Maksimum (Maximum)**: Veri setindeki en büyük değerdir.
*   **Aralık (Range)**: Maksimum ve minimum değerler arasındaki farktır. Veri setinin yayılımını gösterir.
*   **25. Percentil (Q1)**: Verilerin %25'inin altında olduğu değerdir.
*   **75. Percentil (Q3)**: Verilerin %75'inin altında olduğu değerdir.
*   **Interquartile Range (IQR)**: Q3 ve Q1 arasındaki farktır. Veri setinin orta %50'sini ölçer ve uç değerlerden etkilenmez.

Hesaplanan metrikler, kosinüs benzerlik skorları ve insan geri bildirimi skorları için aşağıdaki gibidir:

*   **Ortalama (Mean)**: 0.68
*   **Medyan (Median)**: 0.71
*   **Standart Sapma (Standard Deviation)**: 0.15
*   **Varyans (Variance)**: 0.02
*   **Minimum (Minimum)**: 0.45
*   **Maksimum (Maximum)**: 0.90
*   **Aralık (Range)**: 0.46
*   **25. Percentil (Q1)**: 0.56
*   **75. Percentil (Q3)**: 0.80
*   **Interquartile Range (IQR)**: 0.24

Bu metrikler, veri setinin merkezi eğilimini, yayılımını, uç değerlerini ve dağılımını anlamak için kullanılır.

---

## Summary

## Özet
Bu bölümde, Wikipedia API ve LlamaIndex kullanarak ölçeklenebilir bir bilgi grafiği tabanlı RAG (Retrieval-Augmented Generation) sistemi oluşturmayı keşfettik. Geliştirilen teknikler ve araçlar, veri yönetimi, pazarlama ve organize edilmiş ve erişilebilir veri geri getirmeyi gerektiren her alanda uygulanabilir.

## Önemli Noktalar
*   Wikipedia API ve LlamaIndex kullanarak ölçeklenebilir bir bilgi grafiği tabanlı RAG sistemi oluşturma
*   Veri toplama, işleme ve sorgulama için üç ardışık düzen (Pipeline) oluşturma
*   Derin Göl vektör deposu (Deep Lake vector store) oluşturma ve doldurma
*   Bilgi grafiği indeksli RAG kullanarak sorgulama ve metrikleri değerlendirme

## Ardışık Düzenler (Pipelines)
### Pipeline 1: Veri Toplama
Bu ardışık düzen, Wikipedia'dan içerik almayı otomatikleştirmeye odaklandı. Wikipedia API'sini kullanarak, pazarlama gibi seçilen bir konu temelinde Wikipedia sayfalarından meta veri ve URL'leri toplamak için bir program oluşturduk.

### Pipeline 2: Derin Göl Vektör Deposu Oluşturma
Bu ardışık düzen, Derin Göl vektör deposunu oluşturdu ve doldurdu. Pipeline 1'den alınan veriler gömüldü (embedded) ve Derin Göl vektör deposuna yüklendi (upserted).

### Pipeline 3: Bilgi Grafiği İndeksli RAG
Bu ardışık düzen, LlamaIndex kullanarak bilgi grafiği indeksli RAG'ı tanıttı. Gömülü verilerden otomatik olarak bir bilgi grafiği indeksi oluşturuldu. Bu indeks, farklı bilgi parçaları arasındaki ilişkileri görsel olarak haritalandırarak verilerin anlamsal bir genel bakışını sağladı.

## Kodlar ve Açıklamalar
Aşağıdaki kod örnekleri, Wikipedia API'sini kullanarak veri toplama, Derin Göl vektör deposu oluşturma ve LlamaIndex kullanarak bilgi grafiği indeksli RAG oluşturma işlemlerini gösterir.

### Wikipedia API'sini Kullanarak Veri Toplama
```python
import wikipedia

# Wikipedia API'sini kullanarak veri toplama
def collect_data(topic):
    wikipedia.set_lang("en")
    search_results = wikipedia.search(topic)
    data = []
    for result in search_results:
        try:
            page = wikipedia.page(result)
            data.append({
                "title": page.title,
                "url": page.url,
                "content": page.content
            })
        except wikipedia.exceptions.DisambiguationError:
            pass
    return data

# Veri toplama
topic = "marketing"
data = collect_data(topic)
```
Bu kod, Wikipedia API'sini kullanarak belirli bir konu hakkında veri toplar. `wikipedia` kütüphanesini kullanarak, arama sonuçlarını alır ve her bir sayfanın başlığını, URL'sini ve içeriğini bir liste olarak döndürür.

### Derin Göl Vektör Deposu Oluşturma
```python
import deeplake

# Derin Göl vektör deposu oluşturma
def create_deep_lake_vector_store(data):
    ds = deeplake.dataset("wiki_data")
    ds.create_tensor("text")
    ds.create_tensor("embedding")
    for item in data:
        ds.append({
            "text": item["content"],
            "embedding": get_embedding(item["content"])
        })

# Gömme (embedding) fonksiyonu
def get_embedding(text):
    # Gömme modeli kullanılarak metin gömülür
    # Örneğin, sentence-transformers kütüphanesini kullanabilirsiniz
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode(text)
    return embedding

# Derin Göl vektör deposu oluşturma
create_deep_lake_vector_store(data)
```
Bu kod, Derin Göl vektör deposu oluşturur ve topladığı verileri bu depoya yükler. `deeplake` kütüphanesini kullanarak, bir veri kümesi oluşturur ve metin ve gömme tensörlerini tanımlar.

### LlamaIndex Kullanarak Bilgi Grafiği İndeksli RAG Oluşturma
```python
import llama_index

# LlamaIndex kullanarak bilgi grafiği indeksli RAG oluşturma
def create_knowledge_graph_index(data):
    # Gömme modeli kullanılarak veriler gömülür
    embeddings = [get_embedding(item["content"]) for item in data]
    # LlamaIndex kullanarak bilgi grafiği indeksi oluşturma
    index = llama_index.KnowledgeGraphIndex(embeddings, data)
    return index

# Bilgi grafiği indeksli RAG oluşturma
index = create_knowledge_graph_index(data)

# Sorgulama
query = "What is marketing?"
response = index.query(query)
print(response)
```
Bu kod, LlamaIndex kullanarak bilgi grafiği indeksli RAG oluşturur. Gömme modeli kullanılarak veriler gömülür ve LlamaIndex kullanarak bir bilgi grafiği indeksi oluşturulur. Daha sonra, bir sorgu yapılarak yanıt alınır.

## Sonuç
Bu bölümde, Wikipedia API ve LlamaIndex kullanarak ölçeklenebilir bir bilgi grafiği tabanlı RAG sistemi oluşturmayı keşfettik. Geliştirilen teknikler ve araçlar, veri yönetimi, pazarlama ve organize edilmiş ve erişilebilir veri geri getirmeyi gerektiren her alanda uygulanabilir. Artık gerçek dünya projelerinde bilgi grafiği tabanlı RAG sistemlerini uygulayabilirsiniz.

---

## Questions

## Konu
Bu bölüm, Wikipedia API ve LlamaIndex kullanarak ölçeklenebilir bir bilgi grafiği tabanlı RAG (Retrieval-Augmented Generation) sistemi oluşturmaya odaklanmaktadır.

## Önemli Noktalar
- Bölüm, Wikipedia API ve LlamaIndex kullanarak ölçeklenebilir bir bilgi grafiği tabanlı RAG sistemi oluşturmayı amaçlamaktadır (Building a Scalable Knowledge-Graph-Based RAG System).
- Birinci Pipeline (`Pipeline 1`), Wikipedia'dan API kullanarak doküman toplama ve hazırlamayı içerir (Collecting and Preparing Documents).
- `Pipeline 2`'de Deep Lake, ilişkisel bir veritabanı oluşturmak için kullanılmamaktadır; bunun yerine başka bir amaç için kullanılmaktadır (Not Creating a Relational Database).
- `Pipeline 3`, LlamaIndex'i bir bilgi grafiği indeksi oluşturmak için kullanır (Building a Knowledge Graph Index using LlamaIndex).
- Sistem, belirli bir konuda (örneğin pazarlama) esneklik göstermeden yalnızca tek bir konuyu ele alacak şekilde tasarlanmamıştır (Not Limited to a Single Topic).
- Bölüm, Wikipedia sayfalarından URL'leri ve meta verileri nasıl alacağınızı açıklar (Retrieving URLs and Metadata).
- Açıklanan pipeline'ları çalıştırmak için bir GPU gerekli değildir (Not Requiring a GPU).
- Bilgi grafiği indeksi, veri parçaları arasındaki ilişkileri görsel olarak haritalandırır (Visually Mapping Out Relationships).
- Bilgi grafiği indeksini sorgulamak için her adımda insan müdahalesine gerek yoktur (Not Requiring Human Intervention at Every Step).

## İlgili Kodlar ve Açıklamalar
İlgili kodlar ve açıklamalar aşağıdaki gibidir:
### Wikipedia API ile Doküman Toplama
```python
import requests

def get_wikipedia_page(title):
    url = f"https://en.wikipedia.org/w/api.php?action=parse&page={title}&format=json"
    response = requests.get(url)
    return response.json()

# Kullanımı
title = "Artificial Intelligence"
result = get_wikipedia_page(title)
print(result)
```
Bu kod, Wikipedia API'sini kullanarak belirli bir sayfanın içeriğini JSON formatında almak için kullanılır.

### LlamaIndex ile Bilgi Grafiği İndeksi Oluşturma
```python
from llama_index import KnowledgeGraphIndex

def create_knowledge_graph_index(documents):
    index = KnowledgeGraphIndex(documents)
    return index

# Kullanımı
documents = [...]  # Doküman listesi
index = create_knowledge_graph_index(documents)
print(index)
```
Bu kod, LlamaIndex kütüphanesini kullanarak bir dizi dokümandan bilgi grafiği indeksi oluşturur.

### Deep Lake Kullanımı
```python
import deeplake

def create_deeplake_dataset():
    ds = deeplake.dataset("path/to/dataset")
    return ds

# Kullanımı
ds = create_deeplake_dataset()
print(ds)
```
Bu kod, Deep Lake kütüphanesini kullanarak bir veri seti oluşturur veya erişir.

## Cevaplar
1. Does the chapter focus on building a scalable knowledge-graph-based RAG system using the Wikipedia API and LlamaIndex? 
   - **Evet (Yes)**
2. Is the primary use case discussed in the chapter related to healthcare data management? 
   - **Hayır (No)**
3. Does Pipeline 1 involve collecting and preparing documents from Wikipedia using an API? 
   - **Evet (Yes)**
4. Is Deep Lake used for creating a relational database in Pipeline 2? 
   - **Hayır (No)**
5. Does Pipeline 3 utilize LlamaIndex to build a knowledge graph index? 
   - **Evet (Yes)**
6. Is the system designed to only handle a single specific topic, such as marketing, without flexibility? 
   - **Hayır (No)**
7. Does the chapter describe how to retrieve URLs and metadata from Wikipedia pages? 
   - **Evet (Yes)**
8. Is a GPU required to run the pipelines described in the chapter? 
   - **Hayır (No)**
9. Does the knowledge graph index visually map out relationships between pieces of data? 
   - **Evet (Yes)**
10. Is human intervention required at every step to query the knowledge graph index? 
    - **Hayır (No)**

---

## References

## Vikipedi API ve Ağ Görselleştirme
Vikipedi API, geliştiricilerin Vikipedi verilerine erişmelerini sağlayan bir arabirimdir (interface). Bu API, çeşitli programlama dilleri kullanılarak erişilebilir. Bu makalede, Python kullanarak Vikipedi API'sine nasıl erişileceği ve elde edilen verilerin nasıl görselleştirileceği anlatılacaktır.

## Kullanılan Kütüphaneler ve Araçlar
- `wikipediaapi`: Vikipedi API'sine erişmek için kullanılan Python kütüphanesidir.
- `pyvis`: Ağ görselleştirme için kullanılan Python kütüphanesidir.

## Vikipedi API'sine Erişim
Vikipedi API'sine erişmek için `wikipediaapi` kütüphanesini kullanacağız. Bu kütüphaneyi kullanmak için öncelikle kurulumunu gerçekleştirmeliyiz.

### Kurulum
```bash
pip install wikipediaapi
```
## Kod Örneği
Aşağıdaki kod örneğinde, `wikipediaapi` kütüphanesini kullanarak Vikipedi'den bir sayfa nasıl çekilir ve içeriği nasıl işlenir, gösterilmektedir.

```python
import wikipediaapi

# Vikipedi API nesnesini oluştur
wiki = wikipediaapi.Wikipedia('en')  # 'en' İngilizce Vikipedi için

# Sayfa nesnesini oluştur
page = wiki.page('Python_(programming_language)')

# Sayfa var mı kontrol et
if page.exists():
    print(page.title)  # Sayfa başlığını yazdır
    print(page.summary)  # Sayfa özetini yazdır
else:
    print("Page not found.")
```
### Kod Açıklaması
- `wikipediaapi.Wikipedia('en')`: İngilizce Vikipedi için bir API nesnesi oluşturur. Burada `'en'` parametresi dil kodudur (language code).
- `wiki.page('Python_(programming_language)')`: Belirtilen başlığa sahip sayfa nesnesini oluşturur.
- `page.exists()`: Sayfanın var olup olmadığını kontrol eder.
- `page.title` ve `page.summary`: Sayfa başlığını ve özetini döndürür.

## Ağ Görselleştirme
Elde edilen verileri görselleştirmek için `pyvis` kütüphanesini kullanacağız. Bu kütüphane, etkileşimli ağ görselleştirmeleri oluşturmayı sağlar.

### Kurulum
```bash
pip install pyvis
```
## Kod Örneği
Aşağıdaki kod örneğinde, basit bir ağ nasıl oluşturulur ve görselleştirilir, gösterilmektedir.

```python
from pyvis.network import Network

# Ağ nesnesini oluştur
net = Network()

# Düğümler ekle
net.add_node(1, label='Node 1')
net.add_node(2, label='Node 2')
net.add_node(3, label='Node 3')

# Kenarlar ekle
net.add_edge(1, 2)
net.add_edge(2, 3)
net.add_edge(3, 1)

# Ağ'ı kaydet
net.save_graph('example.html')
```
### Kod Açıklaması
- `Network()`: Boş bir ağ nesnesi oluşturur.
- `add_node()`: Ağa düğüm ekler. İlk parametre düğüm ID'si, `label` parametresi düğüm etiketidir.
- `add_edge()`: Ağa kenar ekler. İki düğüm arasındaki bağlantıyı tanımlar.
- `save_graph()`: Oluşturulan ağı HTML dosyası olarak kaydeder. Bu dosya, web tarayıcısında açılarak etkileşimli olarak görüntülenebilir.

## Önemli Noktalar
- Vikipedi API'sine erişmek için uygun dil kodunu (`'en'`, `'tr'`, vs.) belirtmelisiniz.
- `wikipediaapi` ve `pyvis` kütüphanelerinin kurulumunu yapmalısınız.
- Ağ görselleştirme için `pyvis` kütüphanesini kullanarak etkileşimli grafikler oluşturabilirsiniz.

## Kaynaklar
- [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page)
- [wikipediaapi GitHub repository](https://github.com/martin-majlis/Wikipedia-API)
- [PyVis Network](https://pyvis.readthedocs.io/en/latest/)

---

## Further reading

## Bilgi Grafikleri (Knowledge Graphs) Hakkında Daha Fazla Okuma

Bilgi grafikleri, varlıklar (entities) arasındaki ilişkileri temsil eden bir veri yapısıdır (data structure). Bu yapı, varlıkların özelliklerini ve birbirleriyle olan ilişkilerini tanımlar. Bilgi grafikleri, doğal dil işleme (Natural Language Processing, NLP), yapay zeka (Artificial Intelligence, AI) ve veri entegrasyonu (Data Integration) gibi alanlarda yaygın olarak kullanılmaktadır.

## Bilgi Grafikleri Tanımı

Bilgi grafikleri, düğümler (nodes) ve kenarlar (edges) olarak temsil edilir. Düğümler varlıkları, kenarlar ise bu varlıklar arasındaki ilişkileri temsil eder. Örneğin, bir film bilgisi grafikte bir düğüm olarak temsil edilebilir ve bu düğümün özellikleri filmün adı, yönetmeni ve oyuncuları olabilir. Kenarlar ise bu filmün yönetmeni ve oyuncuları arasındaki ilişkileri temsil edebilir.

## Bilgi Grafikleri Kullanım Alanları

*   Doğal dil işleme (NLP)
*   Yapay zeka (AI)
*   Veri entegrasyonu (Data Integration)
*   Öneri sistemleri (Recommendation Systems)
*   Bilgi tabanlı sistemler (Knowledge-based Systems)

## Bilgi Grafikleri Oluşturma

Bilgi grafikleri oluşturmak için çeşitli yöntemler vardır. Bu yöntemler arasında:

*   Varlık tanıma (Entity Recognition)
*   İlişki çıkarma (Relation Extraction)
*   Veri entegrasyonu (Data Integration)

### Örnek Kod

Aşağıdaki örnek kod, Python programlama dilinde NetworkX kütüphanesini kullanarak basit bir bilgi grafiği oluşturmayı gösterir:
```python
import networkx as nx
import matplotlib.pyplot as plt

# Boş bir grafik oluştur
G = nx.Graph()

# Varlıkları ekle
G.add_node("Film")
G.add_node("Yönetmen")
G.add_node("Oyuncu")

# İlişkileri ekle
G.add_edge("Film", "Yönetmen", label="Yönetmenlik")
G.add_edge("Film", "Oyuncu", label="Oyuncu")

# Grafiği çiz
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue')
labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
```
Bu kod, "Film", "Yönetmen" ve "Oyuncu" adlı varlıkları ve bu varlıklar arasındaki "Yönetmenlik" ve "Oyuncu" ilişkilerini temsil eden bir grafik oluşturur.

### Kod Açıklaması

*   `import networkx as nx`: NetworkX kütüphanesini içe aktarır.
*   `import matplotlib.pyplot as plt`: Matplotlib kütüphanesini içe aktarır.
*   `G = nx.Graph()`: Boş bir grafik oluşturur.
*   `G.add_node()`: Varlıkları grafiğe ekler.
*   `G.add_edge()`: İlişkileri grafiğe ekler.
*   `pos = nx.spring_layout(G)`: Grafiğin düzenini belirler.
*   `nx.draw()`: Grafiği çizer.
*   `nx.draw_networkx_edge_labels()`: Kenar etiketlerini çizer.

## Kaynaklar

*   Hogan, A., Blomqvist, E., Cochez, M., et al. Knowledge Graphs. arXiv:2003.02320

Bu kaynak, bilgi grafikleri hakkında detaylı bir inceleme sunmaktadır.

---

## Join our community on Discord

## Discord Topluluğumuza Katılın
Packt linki üzerinden Discord platformunda yazar ve diğer okuyucular ile tartışmalara katılmak için topluluğumuza katılabilirsiniz: https://www.packt.link/rag

## Önemli Noktalar
- Yazar ve diğer okuyucular ile tartışmalara katılma imkanı
- Discord platformunda etkileşimli sohbet
- Packt linki üzerinden erişim

## Teknik Detaylar
Paragrafda teknik bir kod veya detay bulunmamaktadır. Ancak Discord'a katılma linki verilmiştir: `https://www.packt.link/rag`

## Kod Kullanımı
Bu bölümde kod bulunmamaktadır.

## Ek Bilgiler
- Discord, topluluklarla etkileşimli sohbet için popüler bir platformdur (Platform).
- Packt, teknik kitaplar ve kaynaklar sunan bir yayıncıdır (Publisher).

## İlgili Linkler
- https://www.packt.link/rag

## Markdown Kullanımı
Yukarıdaki başlıklar markdown formatında (`## Yazı`) düzenlenmiştir.

## Kod Açıklaması
Bu metinde herhangi bir kod örneği bulunmamaktadır.

## İthal Edilen Kütüphaneler (Import)
Bu metinde herhangi bir kod veya import ifadesi bulunmamaktadır.

## Genel Bilgiler
Discord topluluğuna katılmak için verilen linki takip edebilirsiniz. Bu, yazar ve diğer okuyucular ile tartışmalara katılmanıza olanak tanır.

---

