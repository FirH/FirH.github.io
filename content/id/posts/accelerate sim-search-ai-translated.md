+++ date = '2026-01-02T08:50:21+07:00' draft = false title = 'Mempercepat Proses Similarity Search Melalui Distribusi Tugas, Memory Coalescing, dan Shared Memory' +++

# Mempercepat Proses Similarity Search Melalui Distribusi Kerja, Memory Coalescing, dan Shared Memory

## Pendahuluan

Dalam masa maraknya penggunaan Large Language Model (LLM) dan Retrieval-Augmented Generation (RAG), _similarity search_ adalah operasi mendasar seperti pada sistem pencarian informasi berbasis semantik, sistem rekomendasi, atau sebuah pipeline RAG. Operasi ini meliputi pencarian vektor dengan kemiripan paling tinggi pada sebuah basis data terhadap vektor masukan / kueri dari pengguna. Bagaimana jika basis data yang kita gunakan memiliki jutaan vektor? Performa menjadi salah pertimbangan penting dalam pembangunan sistem ini.

Dalam artikel ini, saya akan menunjukkan teknik dari optimasi GPU untuk _similarity search_, seperti distribusi kerja dengan _thread_, _memory coalesing_, dan _shared memory_. Artikel ini juga akan menjelaskan implementasi CUDA, demonstrasi peningkatan kinerja, dan menjelaskan alasan dibalik keberhasilan teknik yang digunakan

## Skala Operasi Similarity Search

_Similarity search_ adalah sebuah tugas pencarian objek dengan kemiripan paling tinggi ketika diberikan sebuah masukan _query_. Dalam konteks _embedding_ dari sebuah vektor, proses ini adalah melakukan komputasi berdasarkan metrik kemiripan antara vektor kueri dan sejumlah vektor yang ada pada basis data, hingga ditemukannya hasil dengan nilai metrik kemiripan tertinggi.

Misal terdapat sebuah aplikasi RAG, ketika pengguna memberikan sebuah pernyataan, sistem ini harus:

1. Mengubah pertanyaan menjadi sebuah _vector embedding_
2. Membandingkan _vector embedding_ tersebut dengan _embedding_ dari sejumlah dokumen pada basis data
3. Mengembalikan sejumlah _k_ dokumen paling relevan dari hasil proses
4. Memberikan hasil tersebut pada LLM dalam sistem

Jika jumlah vektor dari basis data adalah jutaan, maka komputasi kemiripan ini dapat menjadi _bottleneck_ yang berat dalam sistem. Pada CPU, memroses 10,000 _query_ terhadap 1 juta basis data vektor melibatkan setidaknya 1.28 juta operasi _dot product_. GPU di sisi lainnya, memiliki _core_ yang dapat digunakan secara paralel untuk melakukan banyak komputasi _dot product_ secara bersamaan. Namun, tetarp perlu diimplementasikan optimasi pola akses memori untuk dapat mempergunakan kapabilitas tersebut dengan baik.

## Persiapan Eksperimen

**Hardware:**

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- CPU: 13th Gen Intel(R) Core(TM) i9-13900HX
- RAM: 32 GB

**Software:**

- Python 3.11.13
- Nvidia Nsight Systems
- Numba 0.63.1
- NumPy 2.3.5

**Dataset:** Dataset yang digunakan adalah SIFT1M, sebuah _dataset_ benchmark untuk tugas _similarity search_:

- **Vektor basis data**: 1,000,000 vektor × 128 dimensi
- **Vektor kueri**: 10,000 vectors × 128 dimensi
- **Acuan kebenaran**: Top-100 _nearest neighbor_ yang telah dikomputasikan sebelumnya
- **Ukuran dataset**: 576.8 MB

## Implementasi GPU Naif

Pada implementasi pertama, setiap thread akan melakukan komputasi kemiripan satu _query_ terhadap seluruh vektor pada basis data:

```python
import numpy as np
import numba
from numba import cuda

@cuda.jit
def similarity_search_non_coalesced(queries, database, results):
    query_idx = cuda.grid(1)
    if query_idx < queries.shape[0]:
        n_db = database.shape[0]
        dim = database.shape[1]
        best_sim = -1e9
        best_idx = -1

        for db_idx in range(n_db):
            sim = 0.0
            for feat_idx in range(dim):
                sim += queries[query_idx, feat_idx] * database[db_idx, feat_idx]

            if sim > best_sim:
                best_sim = sim
                best_idx = db_idx

        results[query_idx, 0] = best_idx
        results[query_idx, 1] = best_sim
```

1. Setiap _thread_ ditugaskan satu _query_ melalui `cuda.grid(1)`
2. Untuk setiap _query_, lakukan iterasi terhadap seluruh basis data vektor
3. Untuk setiap basis data vektor, lakukan komputasi _dot product_
4. Catat nilai kemiripan terbaik dan indeksnya
5. Simpan hasil

Sebelum melakukan benchmark dan mengoptimasi pola akses, kita perlu memahami Memory Coalescing dan Shared Memory.

## Memahami Memory Coalescing

### Hierarki Memori GPU

GPU memiliki beberapa tingkat memori:

- **Global memory**: Memori terbesar dan paling melimpah yang tersedia (GBs), namun paling lambat untuk diakses
- **Shared memory**: Memori on-chip dengan kecepatan jauh lebih cepat daripada global memory
- **Registers**: Tipe memori tercepat yang terletak pada Streaming Multiprocessor

Mengakses global memory dapat menjadi operasi yang mahal, tetapi ketika thread dalam sebuah warp (grup dari 32 thread yang dieksekusi secara bersamaan) mengakses alamat memori yang berurutan, hardware dapat menggabungkan beberapa permintaan memori menjadi satu transaksi.

### Apa itu Memory Coalescing?

Memory coalescing terjadi ketika thread yang berdekatan mengakses lokasi memori yang berdekatan. Alih-alih mengeluarkan 32 permintaan memori terpisah, GPU dapat mengeluarkan satu permintaan yang mengambil semua data yang diperlukan dalam satu transaksi.

Mari kita lihat baris ini pada implementasi naif:

```python
sim += queries[query_idx, feat_idx] * database[db_idx, feat_idx]
```

Sementara thread mengakses `db_idx` yang sama pada waktu yang bersamaan, tata letak memori dari array row-major berarti bahwa `database[db_idx, feat_idx]` dan `database[db_idx, feat_idx+1]` berurutan dalam memori, tetapi `database[db_idx, feat_idx]` dan `database[db_idx+1, feat_idx]` berjauhan (dipisahkan oleh seluruh lebar baris sebesar 128 float).

## Memahami Shared Memory

Shared memory adalah ruang memori yang dapat diakses oleh semua thread dalam satu thread block yang disimpan pada L1 Data Cache dari Streaming Multiprocessor (SM) GPU.

Dalam similarity search, kita dapat menggunakan shared memory untuk:
1. Menyimpan hasil parsial dari setiap thread
2. Melakukan reduksi paralel untuk menemukan kemiripan maksimum
3. Menghindari semua thread menulis ke global memory secara berulang

## Implementasi yang Dioptimasi

Sekarang mari kita lihat bagaimana kita dapat menggabungkan memory coalescing dan shared memory untuk meningkatkan performa:

```python
import numpy as np
import numba
from numba import cuda

@cuda.jit
def similarity_search_true_coalesced(queries, database_T, results):
    query_idx = cuda.blockIdx.x
    thread_idx = cuda.threadIdx.x
    block_size = cuda.blockDim.x

    if query_idx >= queries.shape[0]:
        return

    dim = database_T.shape[0]
    n_db = database_T.shape[1]

    shared_best_sim = cuda.shared.array(256, dtype=numba.float32)
    shared_best_idx = cuda.shared.array(256, dtype=numba.int32)

    local_best_sim = -1e9
    local_best_idx = -1

    for db_idx in range(thread_idx, n_db, block_size):
        sim = 0.0
        for feat_idx in range(dim):
            # COALESCED: database_T[feat_idx, db_idx]
            # Adjacent threads access consecutive memory locations
            # thread 0: db_idx=0, thread 1: db_idx=1, etc.
            sim += queries[query_idx, feat_idx] * database_T[feat_idx, db_idx]

        if sim > local_best_sim:
            local_best_sim = sim
            local_best_idx = db_idx

    shared_best_sim[thread_idx] = local_best_sim
    shared_best_idx[thread_idx] = local_best_idx
    cuda.syncthreads()

    # Reduction in shared memory
    stride = block_size // 2
    while stride > 0:
        if thread_idx < stride:
            if shared_best_sim[thread_idx + stride] > shared_best_sim[thread_idx]:
                shared_best_sim[thread_idx] = shared_best_sim[thread_idx + stride]
                shared_best_idx[thread_idx] = shared_best_idx[thread_idx + stride]
        cuda.syncthreads()
        stride //= 2

    if thread_idx == 0:
        results[query_idx, 0] = shared_best_idx[0]
        results[query_idx, 1] = shared_best_sim[0]
```

## Perbedaan Antara Kedua Implementasi

### 1. Organisasi Thread

**Implementasi Pertama:**

```python
query_idx = cuda.grid(1)  # Setiap thread = satu query
```

**Implementasi Dioptimasi:**

```python
query_idx = cuda.blockIdx.x   # Setiap block = satu query
thread_idx = cuda.threadIdx.x  # Thread bekerja sama pada satu query
```

Pada versi yang dioptimasi, kita menggunakan seluruh block dari thread (256 thread) untuk memproses satu query tunggal.

### 2. Distribusi Kerja

**Implementasi Pertama:**

```python
for db_idx in range(n_db):  # Setiap thread memproses semua vektor
```

**Implementasi Dioptimasi:**

```python
for db_idx in range(thread_idx, n_db, block_size):  # Thread membagi pekerjaan
```

Alih-alih setiap thread memproses semua vektor basis data, thread dalam sebuah block membaginya. Thread 0 memproses vektor 0, 256, 512, ..., sementara thread 1 memproses 1, 257, 513, ..., dan seterusnya.

### 3. Implementasi Memory Coalescing

**Implementasi Pertama:**

```python
database[db_idx, feat_idx]  # Bentuk: (n_vectors, dim)
```

**Implementasi Dioptimasi:**

```python
database_T[feat_idx, db_idx]  # Bentuk: (dim, n_vectors)
```

Dengan mentranspos basis data dari `(n_vectors, dim)` menjadi `(dim, n_vectors)`, kita memastikan bahwa ketika thread yang berdekatan mengakses `database_T[feat_idx, 0]`, `database_T[feat_idx, 1]`, `database_T[feat_idx, 2]`, dll., mereka mengakses lokasi memori yang berurutan.

### 4. Detail Implementasi Shared Memory

Optimasi shared memory diimplementasikan pada baris 36-48:

#### Langkah 1: Deklarasi Shared Memory (baris 18-19)

```python
shared_best_sim = cuda.shared.array(256, dtype=numba.float32)
shared_best_idx = cuda.shared.array(256, dtype=numba.int32)
```

Pertama, kita mengalokasikan dua array di shared memory, satu untuk skor kemiripan, satu untuk indeks. Array ini terlihat oleh semua 256 thread dalam block.
#### Langkah 2: Simpan Hasil Lokal (baris 36-37)

```python
shared_best_sim[thread_idx] = local_best_sim
shared_best_idx[thread_idx] = local_best_idx
```

Setelah setiap thread menemukan kecocokan terbaik di antara vektor basis data yang ditugaskan kepadanya, thread menyimpan hasilnya di shared memory. Thread 0 menulis ke posisi 0, thread 1 ke posisi 1, dll.

#### Langkah 3: Sinkronisasi Thread (baris 38)

```python
cuda.syncthreads()
```

Ini memastikan semua thread telah selesai menulis sebelum kita melanjutkan

#### Langkah 4: Reduksi Paralel (baris 41-48)

```python
stride = block_size // 2
while stride > 0:
    if thread_idx < stride:
        if shared_best_sim[thread_idx + stride] > shared_best_sim[thread_idx]:
            shared_best_sim[thread_idx] = shared_best_sim[thread_idx + stride]
            shared_best_idx[thread_idx] = shared_best_idx[thread_idx + stride]
    cuda.syncthreads()
    stride //= 2
```

Untuk memahami algoritma reduksi paralel, bayangkan 256 nilai yang ingin kita reduksi menjadi satu maksimum:

**Iterasi 1 (stride=128):**

- Thread 0 membandingkan posisi 0 dan 128, menyimpan yang maksimum
- Thread 1 membandingkan posisi 1 dan 129, menyimpan yang maksimum
- ...
- Thread 127 membandingkan posisi 127 dan 255, menyimpan yang maksimum
- Sekarang kita memiliki 128 kandidat

**Iterasi 2 (stride=64):**

- Thread 0 membandingkan posisi 0 dan 64, menyimpan yang maksimum
- Thread 1 membandingkan posisi 1 dan 65, menyimpan yang maksimum
- ...
- Sekarang kita memiliki 64 kandidat

Ini berlanjut (32, 16, 8, 4, 2, 1) hingga thread 0 memegang maksimum global. Alih-alih memiliki satu thread yang secara berurutan memeriksa 256 nilai (256 langkah), reduksi mengurangi langkah menjadi log₂(256) = 8 langkah paralel.

#### Langkah 5: Tulis Hasil Akhir (baris 50-52)

```python
if thread_idx == 0:
    results[query_idx, 0] = shared_best_idx[0]
    results[query_idx, 1] = shared_best_sim[0]
```

Hanya thread 0 yang menulis hasil akhir kembali ke global memory

## Hasil Benchmark

|Implementasi|Waktu (ms)|Queries/detik|Speedup|
|---|---|---|---|
|Non-coalesced|33500.43|299|1.0x|
|Coalesced + Shared Memory|20475.88|488|1.64x|

Kita dapat melihat peningkatan pada Coalesced + Shared Memory berdasarkan waktu yang dibutuhkan untuk menyelesaikan operasi. Membandingkan kedua operasi menggunakan NVIDIA Nsight Systems memberikan lebih banyak detail:

![coalesced](https://claude.ai/images/coalesced.jpg)

![non-coalesced](https://claude.ai/images/non-coalesced.jpg)

Persentase warp yang tidak teralokasi dapat dilihat mencapai 80% pada implementasi pertama. Setelah mendistribusikan beban kerja dan mengimplementasikan teknik optimasi tersebut pada versi coalesced, slot warp yang tersedia telah dimaksimalkan pada SM hingga >99%.

## Kesimpulan

Dalam artikel ini, kami telah mendemonstrasikan dua teknik optimasi GPU:

1. **Memory Coalescing**: Mengorganisir data dan pola akses thread sehingga thread dalam sebuah warp mengakses lokasi memori yang berurutan, mengurangi jumlah transaksi memori hingga satu order of magnitude.
    
2. **Shared Memory**: Menggunakan memori on-chip yang cepat dan dapat diprogram untuk hasil intermediat dan algoritma paralel seperti reduksi, menghindari akses global memory yang mahal.
    

Optimasi ini menghasilkan peningkatan kecepatan 1.64x pada benchmark SIFT1M yang dapat bermanfaat untuk sistem produksi yang memproses jutaan query melalui penghematan biaya. Memahami dan menerapkan optimasi tingkat rendah ini menjadi semakin penting seiring sistem AI terus berkembang.