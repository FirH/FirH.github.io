+++
date = '2026-01-02T08:50:21+07:00'
draft = false
title = 'Mempercepat Proses Similarity Search Melalui Distribusi Tugas, Memory Coalescing, dan Shared Memory'
+++

# Mempercepat Proses Similarity Search Melalui Distribusi Kerja, Memory Coalescing, dan Shared Memory

## Pendahuluan

Similarity search adalah operasi mendasar seperti pada sistem pencarian yang kerap terdapat pada sistem di masa maraknya penggunaan Large Language Model (LLM) atau Retrieval Augmented Generation (RAG), seperti sistem pencarian informasi berbasis semantik, sistem rekomendasi, atau sebuah pipeline RAG. Operasi ini meliputi pencarian vektor dengan kemiripan paling tinggi pada sebuah basis data terhadap vektor masukan / kueri dari pengguna. Bagaimana jika basis data yang kita gunakan memiliki jutaan vektor? Performa menjadi salah pertimbangan penting dalam pembangunan sistem ini.

Dalam artikel ini, saya akan menunjukkan teknik dari optimasi GPU untuk *similarity search*, seperti distribusi kerja dengan *thread*, *memory coalesing*, dan *shared memory*. Artikel ini juga akan menjelaskan implementasi CUDA, demonstrasi peningkatan kinerja, dan menjelaskan alasan dibalik keberhasilan teknik yang digunakan
## Skala Operasi Similarity Search

*Similarity search* adalah sebuah tugas pencarian objek dengan kemiripan paling tinggi ketika diberikan sebuah masukan *query*. Dalam konteks *embedding* dari sebuah vektor, proses ini adalah melakukan komputasi berdasarkan metrik kemiripan antara vektor kueri dan sejumlah vektor yang ada pada basis data, hingga ditemukannya hasil dengan nilai metrik kemiripan tertinggi.

Misal terdapat sebuah aplikasi RAG, ketika pengguna memberikan sebuah pernyataan, sistem ini harus:
1. Mengubah pertanyaan menjadi sebuah *vector embedding*
2. Membandingkan *vector embedding* tersebut dengan *embedding* dari sejumlah dokumen pada basis data
3. Mengembalikan sejumlah *k* dokumen paling relevan dari hasil proses
4. Memberikan hasil tersebut pada LLM dalam sistem

Jika jumlah vektor dari basis data adalah jutaan, maka komputasi kemiripan ini dapat menjadi *bottleneck* yang berat dalam sistem. Pada CPU, memroses 10,000 *query* terhadap 1 juta basis data vektor melibatkan setidaknya 1.28 juta operasi *dot product*. GPU di sisi lainnya, memiliki 
*core* yang dapat digunakan secara paralel untuk melakukan banyak komputasi *dot product* secara bersamaan. Namun, tetarp perlu diimplementasikan optimasi pola akses memori untuk dapat mempergunakan kapabilitas tersebut dengan baik. 
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

**Dataset:**
Dataset yang digunakan adalah SIFT1M, sebuah *dataset* benchmark untuk tugas *similarity search*:
- **Vektor basis data**: 1,000,000 vektor × 128 dimensi
- **Vektor kueri**: 10,000 vectors × 128 dimensi
- **Acuan kebenaran**: Top-100 *nearest neighbor* yang telah dikomputasikan sebelumnya
- **Ukuran dataset**: 576.8 MB
## Implementasi GPU Naif

Pada implementasi pertama, setiap thread akan melakukan komputasi kemiripan satu *query* terhadap seluruh vektor pada basis data:

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

1. Setiap *thread* ditugaskan satu *query* melalui `cuda.grid(1)`
2. Untuk setiap *query*, lakukan iterasi terhadap seluruh basis data vektor
3. Untuk setiap basis data vektor, lakukan komputasi *dot product*
4. Catat nilai kemiripan terbaik dan indeksnya
5. Simpan hasil

Sebelum melakukan *benchmark* dan mengoptimasi pola akses memori, akan dijelaskan secara singkat terkait konsep Memory Coalescing dan Shared Memory.
## Memory Coalescing

### Hierarki Memori GPU

GPU memiliki beberapa tingkat memori:
- **Global memory**: Memori terbesar dan memiliki sumber daya tersedia paling melimpah (GBs), namun paling lambat untuk diakses
- **Shared memory**: Memori *on-chip* dengan kecepatan yang jauh lebih cepat daripada global memory
- **Registers**: Tipe memori tercepat yang terletak pada Streaming Multiprocessor (SM)

Mengakses *global memory* dapat menjadi operasi yang berat dari segi komputasi, namun ketika *thread* dalam sebuah *warp* (kelompok yang terdiri dari 32 *thread* yang dieksekusi secara bersamaan) mengakses alamat memori yang berurutan, GPU dapat menggabungkan beberapa permintaan memori akses tersebut menjadi satu transaksi.
### Apa itu Memory Coalescing?
Memory coalescing terjadi ketika *thread* yang berdekatan mengakes memori yang lokasinya juga berdekatan. Teknik ini membuat GPU dapat mengeluarkan satu permintaan yang mengambil semua data yang diperlukan dalam satu transaksi alih-alih mengeluarkan 32 permintaan akses terpisah

Berikut merupakan contoh implementasi naif Memoy Coalescing:

```python
sim += queries[query_idx, feat_idx] * database[db_idx, feat_idx]
```

Meskipun sejumlah *thread* mengakses `db_idx` di waktu yang sama, susunan *array* *row-major* dari memori berarti `database[db_idx, feat_idx]` dan `database[db_idx, feat_idx+1]` berurutan dari segi memori, namun `database[db_idx, feat_idx]` and `database[db_idx+1, feat_idx]` terpisah satu baris berukuran dengan jarak 128 data bertipe *float* 
## Apa itu Shared Memory?

Shared memory adalah sebuah ruang memori yang dapat diakses oleh semua *thread* dalam satu blok *thread* yang disimpan pada L1 *Data Cache* dari Streaming Multiprocessor (SM) dari GPU. Penggunaan dari *shared memory* pada operasi seperti *similarity search* untuk menghindari keharusan dari semua *thread* untuk menulis hasil operasi ke memori *global* secara berulang

## Implementasi Teroptimasi

Berikut merupakan implementasi gabungan *memory coalescing* dan *shared memory* untuk meningkatkan performa:

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

### 1. Penyusunan Thread
**Implementasi Pertama:**
```python
query_idx = cuda.grid(1)  # Setiap thread memroses satu query
```

**Implementasi Teroptimasi:**
```python
query_idx = cuda.blockIdx.x   # Setiap block memroses satu query
thread_idx = cuda.threadIdx.x  # Thread bekerjasama pada satu query
```

Pada versi teroptimasi, keseluruhan dari *thread* pada satu *block* digunakan untuk memroses satu *query*
### 2. Distribusi Beban Kerja
**Implementasi Pertama:**
```python
for db_idx in range(n_db):  # Setiap thread memroses semua vektor
```

**Implementasi Teroptimasi:**
```python
for db_idx in range(thread_idx, n_db, block_size):  # Threads membagi beban pekerjaan
```

Dibandingkan setiap *thread* memroses semua vektor dari basis data, implementasi teroptimasi membuat setiap *thread*  pada *block* dapat membagi pemrosesan tersebut. Misal, *thread* 0 memroses vektor 0, 256, 512, ..., sementara *thread* 1 memroses vektor 1, 257, 513, ..., dan seterusnya
### 3. Implementasi Memory Coalescing
**Implementasi Pertama:**
```python
database[db_idx, feat_idx]  # Bentuk: (n_vectors, dim)
```

**Implementasi Teroptimasi:**
```python
database_T[feat_idx, db_idx]  # Bentuk: (dim, n_vectors)
```

Transpos yang dilakukan dari `(n_vectos, dim)` menjadi `(dim, n_vectors)` memastikan bahwa *thread* yang berdekatan mengakses lokasi memori yang berurutan
### 4. Detil Implementasi Shared Memory

Optimasi *shared memory* diimplementasikan pada baris 36-48:
#### Langkah 1: Deklarasi Shared Memory (baris 18-19)
```python
shared_best_sim = cuda.shared.array(256, dtype=numba.float32)
shared_best_idx = cuda.shared.array(256, dtype=numba.int32)
```

Pertama, dialokasikan dua *array* di *shared memory*, satu untuk *similarity score*, satu untuk indeks. Jumlah *thread* dari satu *block* yang digunakan dalam *array* ini adalah 256.
#### Step 2: Simpan Hasil Lokal (baris 36-37)
```python
shared_best_sim[thread_idx] = local_best_sim
shared_best_idx[thread_idx] = local_best_idx
```

Setelah setiap *thread* menemukan vektor dengan nilai *similarity* tertinggi di antara vektor basis data yang ditugaskan, maka *thread* tersebut akan menyimpan hasilnya pada *shared memory*.
#### Step 3: Sinkronisasi Thread (baris 38)
```python
cuda.syncthreads()
```

Potongan kode ini memastikan semua thread telah selesai menyimpan hasil masing masing operasi sebelum dilanjutkan ke iterasi berikutnya
#### Step 4: Teknik Reduksi Komputasi Paralel (baris 41-48)
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

Untuk memahami algoritma reduksi paralel, dimisalkan terdapat 256 nilai yang ingin direduksi menjadi 1 nilai maksimum secara global:

**Iterasi 1 (stride=128):**

- Thread 0 membandingkan posisi 0 dan 128, kemudian menyimpan nilai maksimum
- Thread 1 membandingkan posisi 1 dan 129, kemudian menyimpan nilai maksimum
- ...
- Thread 127 membandingkan posisi 127 dan 255, kemudian menyimpan nilai maksimum
- Hingga terdapat 128 kandidat

**Iterasi 2 (stride=64):**

- Thread 0 membandingkan posisi 0 dan 64, kemudian menyimpan nilai maksimum
- Thread 1 membandingkan posisi 1 dan 65, kemudian menyimpan nilai maksimum
- ...
- Hingga terdapat 64 kandidat

Operasi ini berlanjut dengan *stride* yang terus dibagi setengah (32, 16, 8, 4, 2, 1) hingga *thread* 0 memiliki nilai maksimum pada *scope* *global*. Implementasi ini satu *thread* yang awalnuya secara berurutan harus memeriksa 256 nilai, sehingga membutuhkan 256 langkah menjadi menjadi log₂(256) = 8 langkah paralel.
#### Langkah 5: Penulisan Hasil Akhir (baris 50-52)
```python
if thread_idx == 0:
    results[query_idx, 0] = shared_best_idx[0]
    results[query_idx, 1] = shared_best_sim[0]
```

Hanya thread 0 yang menulis hasil akhir kembali ke global memory
## Hasil Benchmark

| Implementasi   | Waktu (ms) | Queries/detik | Speedup |
| -------------- | ---------- | ------------- | ------- |
| Tanpa optimasi | 33500.43   | 299           | 1.0x    |
| Teroptimasi    | 20475.88   | 488           | 1.64x   |
Berikut merupakan perbandingan lebih detil ketika menggunakan NVIDIA Nsight Systems:

![coalesced](/images/coalesced.jpg)

![non-coalesced](/images/non-coalesced.jpg)

Presentase *warp* yang tidak teralokasi dapat dilihat mencapai 80% pada implementasi pertama. Setelah mendistribusikan beban kerja yang diikuti dengan teknik Memory Coalescing dan Shared Memory, *slot warp* yang tersedia telah dimaksimalkan pada SM hingga >99%.
## Kesimpulan

Dalam artikel ini, telah didemonstrasikan teknik optimasi GPU:
1. **Penyusunan Thread dan Distribusi Beban Kerja**: Peralihan beban kerja dari satu *thread* terhadap satu *block of threads*, sehingga menghilangkan sifat *serial* dari pemrosesan satu *query* dengan satu *thread* yang menjadi *bottleneck* dari operasi yang dilakukan 

2. **Memory Coalescing**: Penyusunan data dan pola akses yang dilakukan oleh *thread*, sehingga transaksi yang perlu dilakukan untuk pengambilan data yang awalnya terpisah, dapat dirangkap menjadi satu transaksi.

3. **Shared Memory**: Ruang memori yang dapat diakses oleh semua *thread* dalam suatu *blok thread*, sehingga menghindari keharusan dari semua *thread* untuk menulis hasil operasi secara berulang.

Optimasi ini telah menghasilkan peningkatan kecepatan 1.64x pada benchmark SIFT1M yang dapat bermanfaat untuk penghematan biaya komputasi dari sistem yang juga menerapkan operasi *similarity search*.