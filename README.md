# LLM Fine-Tuning for Task-Oriented Instruction Generation

Berikut ini tahapan Fine-Tuning Open Source LLM untuk *Task Oriented Instruction Generation*.

## 1. Memilih Pretrained Model LLM Open Source

Tahap pertama dalam proses *fine-tuning* adalah memilih model LLM open source yang tepat. Berikut ini kriteria yang perlu diperhatikan:

- **Pilih model disesuaikan dengan use case**  
  Pastikan model yang dipilih relevan dengan kebutuhan spesifik dari sistem yang akan dikembangkan. Oleh karena proses ini berfokus pada *LLM Fine-Tuning for Task-Oriented Instruction Generation*, maka sebaiknya memilih model yang sudah berorientasi pada instruksi, seperti model dengan versi `instruct` karena model jenis ini telah dioptimalkan untuk memahami dan merespons instruksi. 

- **Identifikasi *Resource Training* seperti Kapasitas *Storage*, Komputasi, dan Dataset**  
 Penting untuk memahami kapasitas storage, kemampuan komputasi (GPU), dan ketersediaan data yang mendukung pelatihan model secara optimal. Karena dengan mengindetifikasi *resource training* lebih awal kita dapat memperhitungkan ukuran model yang akan kita lakukan fine-tuning.

- **Pilih Ukuran Model Berdasarkan Jumlah Parameter dan Kapasitas *Resource Training***
    Pilihlah ukuran model yang sesuai dengan kemampuan infrastruktur training yang tersedia. Semakin besar jumlah parameter dalam sebuah model, semakin tinggi pula kebutuhan akan resource komputasi, seperti memori GPU. Oleh karena itu, penting untuk menyesuaikan ukuran parameter model dengan kapasitas sistem agar proses fine-tuning dapat berjalan secara efisien dan stabil.
    
    Sebagai gambaran awal, kebutuhan memori model dapat diperkirakan menggunakan kalkulator dari Hugging Face berikut: [Hugging Face Model Memory Usage Calculator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) atau juga bisa menggunakan [Unsloth](https://github.com/unslothai/unsloth), sebuah framework fine-tuning yang telah dioptimalkan untuk efisiensi memori dan kecepatan, berikut estimasi kebutuhan VRAM berdasarkan ukuran model:

  | Model Parameters | QLoRA (4-bit) VRAM | LoRA (16-bit) VRAM |
  |------------------|--------------------|---------------------|
  | 3B               | 3.5 GB             | 8 GB                |
  | 7B               | 5 GB               | 19 GB               |
  | 8B               | 6 GB               | 22 GB               |
  | 9B               | 6.5 GB             | 24 GB               |
  | 11B              | 7.5 GB             | 29 GB               |
  | 14B              | 8.5 GB             | 33 GB               |
  | 27B              | 22 GB              | 64 GB               |
  | 32B              | 26 GB              | 76 GB               |
  | 40B              | 30 GB              | 96 GB               |
  | 70B              | 41 GB              | 164 GB              |
  | 81B              | 48 GB              | 192 GB              |
  | 90B              | 53 GB              | 212 GB              |
  | 405B             | 237 GB             | 950 GB              |

  Angka-angka di atas merupakan estimasi dan dapat sedikit bervariasi tergantung pada konfigurasi, tokenizer, dan panjang input saat training.

- **Pertimbangkan Memilih Model Terbaru dan Evaluasi Berdasarkan Dataset Instruksi** 
  Untuk mendapatkan performa dan kapabilitas terbaik, disarankan untuk memilih model yang terbaru (*latest release*), karena umumnya membawa peningkatan dalam hal efisiensi dan pemahaman konteks.
  Dalam konteks *LLM Fine-Tuning for Task-Oriented Instruction Generation*, model yang telah dilatih atau dioptimalkan pada data instruksi sangat direkomendasikan yaitu model versi `instruct`, pertimbangkan melakukan benchmarking awal menggunakan dataset berbasis instruksi seperti [IFEval](https://huggingface.co/datasets/ise-uiuc/IFEval) untuk menilai kemampuan dasar model dalam mengikuti instruksi sebelum dilakukan fine-tuning lebih lanjut.

### Pemilihan Kandidat Model

Berdasarkan kriteria yang sudah dijelaskan diatas, dengan mempertimbangkan batas umum infrastruktur training komersial yang tersedia di pasaran saat ini, yang umumnya menyediakan VRAM antara 24–32GB, maka pemilihan kandidat model LLM open-source harus disesuaikan agar proses fine-tuning stabil, efisien, Tidak kehabisan memori VRAM dan dapat berjalan dengan batch size minimal dan panjang input (sequence length) standar, misalnya 2048–4096 token.

Oleh karena itu, model dengan ukuran parameter `maksimal 8B` dipilih sebagai batas optimal. Berdasarkan tabel estimasi kebutuhan VRAM dengan metode QLoRA, ukuran model hingga 8B masih sangat aman untuk dilatih di GPU 24GB.

Berikut ini adalah kandidat model open-source yang telah instruction-tuned dan memiliki performa baik berdasarkan benchmark `IFEval`:

| Model     | Params | Versi        | IFEval |
|-----------|------------------|--------------|-------------------------|
| LLaMA 3.1 | 8B               | Instruct     | 75.0                    |
| Gemma 3   | 12B              | IT           | 80.2                    |
| Qwen 2.5  | 7B               | Instruct     | 71.2                    |
| Qwen 2.5  | 14B              | Instruct     | 81.0                    |
| Qwen3     | 4B               | Instruct     | 81.2                    |
| **Qwen3** | **8B**           | **Instruct** | **83.0**                |

Source: [Qwen3-Technical Report](https://arxiv.org/pdf/2505.09388)

Dari perbandingan di atas, Qwen3-8B-Instruct menjadi pilihan terbaik karena:

- Skor tertinggi (83.0) pada IFEval, menunjukkan bahwa model dengan baik dalam memahami dan merespons instruksi prosedural yang kompleks.
- Masuk dalam batas konsumsi VRAM 24GB secara efisien saat menggunakan QLoRA
- Telah instruction-tuned secara resmi oleh tim Qwen (Alibaba), memiliki dukungan tokenizer multilingual, termasuk Bahasa Indonesia, dan tersedia dalam format yang kompatibel untuk inferensi seperti GGUF.
- Memiliki dokumentasi lengkap, ekosistem LoRA, dan kompatibel dengan framework seperti `transformers`, `peft`, dan `unsloth`.

Dengan demikian, Qwen3-8B-Instruct merupakan kandidat paling ideal dan realistis untuk proyek *LLM Fine-Tuning for Task-Oriented Instruction Generation*, baik dari sisi performa, fleksibilitas, maupun ketersediaan resource training di lingkungan umum.


## 2. Perancangan dan Persiapan Dataset

- **Data yang digunakan**

  Untuk melakukan fine-tuning model LLM agar mampu menghasilkan instruksi prosedural yang sistematis berdasarkan user intent, jenis data yang diperlukan adalah:

  - Prompt berupa pertanyaan atau perintah dari user yang merepresentasikan user intent.
  - Output target berupa langkah-langkah prosedural yang jelas, terstruktur, dan numerik.
  
  Format data dapat disusun dalam bentuk `JSON` maupun `CSV`, yang umumnya terdiri dari dua bagian utama:
  - Kolom `prompt`: untuk input dari user,
  - Kolom `output`: untuk langkah-langkah prosedural sebagai target model.

  Contoh data dalam format JSON:
  
  ```json
  {
    "prompt": "How do I reset my password in the e-commerce 'xx' mobile app?",
    "output": "1. Open the app\n2. Go to the login page\n3. Tap on 'Forgot Password'\n4. Enter your email\n5. Tap 'Submit' to receive reset link\n6. Check your email\n7. Set a new password"
  }
  ```

  Dengan format ini memudahkan pada tahap konversi ke struktur `chat-template` atau `prompt-template` tokenizer data sebelum dilakukan proses fine-tuning pretrained model.

- **Proses Pengumpulan dan Anotasi Dataset**
  
  Berikut ini beberapa langkah-langkah dalam proses:

    - Identifikasi Sumber Data
      
      Pengumpulan data dari sumber-sumber yang relevan dengan penggunaan aplikasi atau sistem berbasis prosedur, seperti Dokumentasi FAQ, SOP (Panduan Penggunaan) atau help center yang umumnya mempunyai format yang mengandung `user intent` dan `structured response`.

    - Anotasi manual

      Jika datanya berasal dari sumber mentah (misalnya forum atau percakapan tidak terstruktur), lakukan anotasi manual untuk:
      
      - Menyusun ulang jawaban menjadi bentuk langkah prosedural.
      - Menghapus informasi yang tidak relevan.
      - Menyederhanakan kalimat panjang menjadi poin-poin singkat.
      - Normalisasikan gaya bahasa yang formal, lugas, dan mudah dipahami.
      - Terapkan penomoran eksplisit untuk langkah-langkah (1, 2, 3...).

    - Synthetic data generation: 
      
      Opsi yang dapat digunakan ketika jumlah data terbatas, atau untuk memperkaya variasi prompt yang memiliki maksud instruksi serupa. Teknik ini memanfaatkan LLM untuk menghasilkan berbagai bentuk pertanyaan yang berbeda namun mengarah pada prosedur yang sama. Dengan melakukan data augmentation semacam ini, proses fine-tuning menjadi lebih efektif dalam memahami beragam cara pengguna menyampaikan intent yang sama. Dengan cara ini, setiap prompt akan memiliki beberapa variasi, misalnya 3–5 variasi.

    - Validasi dan Kualitas

      Libatkan `Domain Expert` untuk mengevaluasi kesesuaian antara prompt dan output. Proses validasi dapat dibantu dengan sistem sederhana seperti review-and-approve untuk memastikan konsistensi dan kualitas data yang dihasilkan.

- **Langkah Preprocessing Dataset**

  - Format Dataset
    
    Pastikan dataset yang digunakan berupa JSON atau CSV yang terdiri dari dua bagian utama yaitu `prompt` dan `output`.

    ```json
    {
      "prompt": [PROMPT],
      "output": [OUTPUT]
    }
    ```

  - Konversi ke Format Chat
  
    Ubah struktur data menjadi format percakapan berbasis peran (role), agar sesuai dengan kebutuhan model berdasarkan format dari ChatML:

    ```
    [
      { "role": "user", "content": [PROMPT] },
      { "role": "assistant", "content": [OUTPUT] }
    ]
    ```
  - Terapkan Chat Template Model

     Gunakan `tokenizer.apply_chat_template()` (bawaan dari tokenizer model) untuk mengubah format chat menjadi prompt teks yang siap ditokenisasi:

     Contoh hasil dari chat template tokenizer model Qwen3:

      ```
      <|im_start|>user
      <prompt>
      <|im_end|>
      <|im_start|>assistant
      <output>
      <|im_end|>
      ```
  - Hitung Panjang Token
    
    Lakukan tokenisasi dan hitung panjang token pada setiap data untuk mengetahui distribusi dan mendeteksi potensi *truncation* akibat max_length. Langkah ini membantu:
    - Menentukan nilai max_length yang tepat pada inisialisasi `Training Argument`, 
    - Menghindari kehilangan bagian output karena terlalu panjang, yang dapat menyebabkan model gagal mempelajari data.
    - Estimasi Resource dengan menyesuaikan batch size dengan kapasitas VRAM, dimana semakin besar max_length maka akan semakian besar penggunaan VRAM.

    Opsi jika data terlalu panjang:
    - Truncation: Potong bagian yang kurang penting.
    - Sliding window: Potong menjadi beberapa bagian bertumpuk.
    - Filtering: Singkirkan data ekstrem jika jumlahnya kecil.

  - Splitting Dataset 
    
    Dataset akan dibagi ke dalam dua kelompok, yaitu train dan eval, berdasarkan variasi dari setiap prompt. Sebagai contoh, jika satu prompt memiliki tiga variasi pertanyaan, maka:
    - 2 variasi dimasukkan ke dalam data train
    - 1 variasi sisanya dimasukkan ke dalam data eval

    Tujuan dari pembagian ini adalah untuk melatih model agar mampu mengenali dan memahami prompt baru yang belum pernah dilihat saat training, tetapi masih berada dalam konteks prompt yang sama.
    
    Dengan pendekatan ini, proses fine-tuning diharapkan dapat membuat model:
    - Lebih tangguh terhadap perbedaan formulasi bahasa,
    - Tetap mampu memberikan instruksi prosedural yang akurat dan relevan
    - Meningkatkan kemampuan generalisasi terhadap variasi ekspresi pengguna


  Setelah data dilakukan formatting berdasarkan chat template dari tokenizer model serta sudah mengetahui panjang token dan menetapkan max_length serta splitting dataset, Maka data siap digunakan dalam pipeline training.

- **Self-correction / Edge Cases**
  
  Beberapa langkah untuk menangani potensi masalah dalam dataset:

  - Data Imbalance
    
    Tambahkan variasi pada konteks prompt yang kurang menggunakan synthetic data generation atau paraphrasing, dan lakukan penyamaan distribusi antar prompt.

  - Informasi Sensitif

    Hapus atau samarkan informasi pribadi (misal: email, nama, nomor HP) menggunakan placeholder seperti [EMAIL], [NAME]. Manfaatkan tools PII Detector atau NER agar informasi sensitif data pribadi tidak masuk dalam data training.
    
  - Diversity & Generalization
  
    Pastikan tiap prompt memiliki beberapa variasi prompt yang berbeda, bisa memanfaatkan synthetic data generation atau paraphrasing. Bagi ke dalam train dan eval agar model belajar mengenali pola baru dalam konteks yang sama.
  
  - Edge Cases
    Filter data yang terlalu panjang, ambigu, atau tidak relevan. Lakukan validasi manual pada kasus khusus bila diperlukan dengan melibatkan annotator atau `Domain Expert`.


## 3. Strategi *Fine-Tuning*

- **Pendekatan Fine-Tuning: QLoRA (Quantized LoRA)**

  Untuk titik awal memilih pendekatan fine-tuning menggunakan PEFT (Parameter-Efficient Fine-Tuning), tepatnya QLoRA (Quantized Low Rank Adapter) yang memungkinkan fine-tuning pretrained model LLM dengan efisien, baik dari sisi memori maupun waktu pelatihan, sehingga sangat cocok untuk infrastruktur terbatas seperti GPU 24GB.

  Konsep QLoRA:

  QLoRA merupakan gabungan dari dua teknik utama yaitu LoRA dan Quantization.

  - LoRA (Low-Rank Adapter)
    
    Alih-alih melatih seluruh bobot model (seperti pada full fine-tuning), LoRA hanya menambahkan layer adapter kecil pada bagian tertentu dari model.
    Selama fine-tuning, hanya lapisan adapter inilah yang diperbarui, sementara parameter asli model tetap dibekukan. Teknik ini dapat:
    
    - Mengurangi jumlah parameter yang dilatih secara drastis.
    - Menghemat resource pelatihan.
    - Mempercepat waktu training.

  - Quantization (Kuantisasi 4-bit)

    Berbeda dari LoRA biasa, QLoRA menggunakan model dasar yang telah dikuantisasi ke dalam representasi 4-bit. Dengan presisi rendah ini, konsumsi memori dan beban komputasi dapat ditekan secara signifikan.
    Hasilnya:
    
    - Model besar (seperti 7B hingga 14B) dapat di-finetune hanya dengan GPU 24GB.
    - Performa tetap kompetitif jika dibandingkan dengan fine-tuning model full-precision.

- **Framework yang Digunakan: Unsloth**

  Untuk mendukung fine-tuning ini, saya menggunakan [Unsloth](https://github.com/unslothai/unsloth), sebuah framework open-source yang dibangun di atas HuggingFace Transformers. Unsloth berfungsi sebagai "accelerator plug-in" yang membuat proses fine-tuning LLM jauh lebih cepat dan efisien secara memori. Keunggulan Unsloth:
  - 2× lebih cepat, 60–70% lebih hemat VRAM dibanding HF+PEFT biasa, 0% akurasi loss.
  - Integrasi Plug-and-play dengan `transformers`, `PEFT`,  `trl`.


- **Key Finetuning Hyperparameter untuk LoRA Config**


  | **Hyperparameter**     | **Fungsi**                                                                                                                                             | **Rekomendasi**                                              |
  |------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
  | **LoRA Rank (`r`)**    | Mengontrol jumlah parameter yang dapat dilatih dalam matriks adaptor LoRA. Rank yang lebih tinggi meningkatkan kapasitas, tetapi juga penggunaan memori. | 8, 16, 32, 64, 128 Disarankan: **16 atau 32**                    |
  | **LoRA Alpha**         | Mengatur intensitas adaptasi. Nilai ini mempengaruhi seberapa besar pengaruh fine-tuning terhadap model utama. biasanya proporsional terhadap `r`.                                                              | Sama dengan `r`, atau `r * 2` (heuristik umum)                         |
  | **Target Modules**     | Menentukan bagian model mana yang ingin diterapkan LoRA — bisa hanya attention, hanya MLP, atau keduanya. Salah pilih bisa membuat adaptasi tidak efektif.                                              | Disarankan target semua: <br>**q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj** |

- **Key Finetuning Hyperparameter untuk Training Argument Config**



    | Hyperparameter               | Fungsi                                                                                                                                                                                                                  | Rekomendasi                                                                                  |
  |-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
  | `learning_rate`             | Mengontrol seberapa besar bobot model disesuaikan pada setiap langkah pelatihan. <br>• **Terlalu tinggi**: cepat konvergen tapi bisa tidak stabil. <br>• **Terlalu rendah**: stabil, tapi bisa lambat atau overfitting. | `2e-4` untuk LoRA/QLoRA                                                                     |
  | `num_train_epochs`          | Menentukan berapa kali model melihat seluruh dataset. <br>• Terlalu banyak → risiko overfitting. <br>• Terlalu sedikit → underfitting.                                             | Umumnya: **1–3 epoch** saja                                                                  |
  | `per_device_batch_size`     | Ukuran batch per GPU. Ukuran besar mempercepat training, tapi makan VRAM. Ukuran kecil lebih stabil.                                                                              | Sesuaikan dengan kapasitas GPU                                                               |
  | `gradient_accumulation_steps` | Mengakumulasi gradient dari beberapa mini-batch sebelum update model, untuk menyimulasikan batch besar tanpa kehabisan memori.<br>**Effective batch size** = `batch_size × gradient_steps × device` | Disesuaikan agar mencapai batch size ideal tanpa melebihi VRAM                               |
  | `max_length`                | Panjang maksimum token input + output. Perlu disesuaikan dengan distribusi panjang data agar tidak terpotong.                                                                     | Cek rata-rata token saat preprocessing                                                       |
  | `warmup_steps`              | Meningkatkan learning rate secara bertahap di awal training untuk stabilitas awal. Mencegah lonjakan loss atau divergence di iterasi awal.                                        | Sekitar **5–10%** dari total langkah pelatihan                                               |
  | `lr_scheduler_type`         | Menentukan bagaimana learning rate berubah selama training.                                                                                                                        | **linear**                                                                                    |
  | `weight_decay`       | Regularisasi tambahan untuk menghukum bobot yang terlalu besar, menjaga model agar tidak overfitting.                                                    | **0.01** (disarankan), maksimal **0.1**                                |


- **Potensial Issue Selama Fine-tuning dan Strategi Mitigasi**

  - Keterbatasan Resource Komputasi (GPU)

    Solusi:

    - Gunakan teknik PEFT seperti QLoRA untuk melatih hanya sebagian kecil parameter dan menggunakan base model kuantisasi 4-bit.
    - Gunakan framework efisien seperti Unsloth yang mengurangi kebutuhan memori dan mempercepat proses pelatihan.
    - Kurangi max_length input, token input yang panjang akan memengaruhi konsumsi VRAM.
    - Atur nilai parameter `bath_size`, lakukan kombinasi percobaan nilai hingga optimal.

  - Overfitting

    Keadaan dimana model terlalu kaku terhadap data training sehingga ketika dihadapkan pada data yang belum pernah dilihatnya akurasinya kecil. Cara mitigasi nya adalah dengan menggunakan beberapa parameter tuning seperti:
    
    - Atur Ulang `Learning Rate`, nilai yang terlalu tinggi dapat menyebabkan overfitting, 
    - Kurangi Jumlah `Epoch`, batasi nilaihanya 1, 2, atau 3 epoch untuk mencegah model terlalu menghafal data.
    - Tingkatkan `weight_decay`, tambahkan "rem" penalti pada bobot yang terlalu besar, sehingga model belajar lebih hati-hati dan tidak "menghafal" data. Nilai umum: 0.01 atau 0.1.
    - Tingkatkan `batch_size` atau `gradient_accumulation_steps`, Ukuran batch yang lebih besar atau akumulasi gradien bisa membantu memperhalus pembelajaran model dan mencegah overfitting.
    - Gunakan Evaluasi + Early Stopping, aktifkan evaluasi saat training dan hentikan jika loss pada data validasi meningkat selama beberapa langkah ini mencegah model terus belajar saat performa menurun.

  - Underfitting

    Keadaan dimana model tidak bisa generalisasi data selama training sehingga nilai akurasinya kecil. Solusi:
    
    - Sesuaikan `Learning Rate`, tingkatkan learning rate untuk mempercepat konvergensi.
    - Tambahkan Jumlah `Epoch`, Lanjutkan pelatihan dalam lebih banyak epoch agar model memiliki kesempatan belajar lebih dalam.
    - Tingkatkan `LoRA Rank (r)` dan `Alpha`, nilai rank (jumlah dimensi matriks adaptor yang dilatih) sebaiknya setara atau lebih kecil dari nilai alpha. Untuk model yang lebih kecil atau dataset yang lebih kompleks, gunakan nilai rank yang lebih besar. 

  - Catastrophic Forgetting

      Model kehilangan pengetahuan awal karena terlalu fokus pada data fine-tuning baru. Solusi:

      - Gunakan teknik finetuning PEFT seperti LoRA/QLoRA, karena pada teknik ini hanya lapisan adapter yang akan diperbarui sementara parameter asli model tetap dibekukan sehingga tetap mempertahankan pengetahuan sebelumnya.


## 4. Evaluasi dan Benchmarking

- **Quantitative Metrics**

  - ROUGE-L
    
    Mengukur kemiripan berdasarkan urutan kata terpanjang yang sama antara prediksi dan referensi, yang disebut LCS (Longest Common Subsequence). 
    
    ROUGE-L cocok untuk mengevaluasi instruksi prosedural karena memperhatikan urutan langkah.

    Ambang Batas Interpretasi ROUGE-L F1:

    | Skor ROUGE-L F1 | Interpretasi                        |
    | --------------- | ----------------------------------- |
    | **> 60%**       | Sangat mirip, langkah-langkah cocok |
    | **40–60%**      | Cukup mirip, urutan agak berbeda    |
    | **< 40%**       | tidak sesuai urutan |

    
  - METEOR

    Mengukur kemiripan semantik antara prediksi dan referensi. METEOR mencocokkan kata-kata menggunakan 4 strategi:
    
    - Exact Match – kata identik
    - Stem Match – kata yang memiliki akar kata sama (run vs running)
    - Synonym Match – sinonim dengan WordNet (buy vs purchase)
    - Paraphrase Match – frasa berbeda, tapi makna setara

    METEOR cocok untuk mengevaluasi kejelasan dan substansi jawaban meskipun strukturnya berbeda.

    | Skor METEOR | Interpretasi                               |
    | ----------- | ------------------------------------------ |
    | **> 60%**   | Sangat baik – substansi dan semantik cocok |
    | **45–60%**  | Baik – cukup mirip, masih relevan          |
    | **< 45%**   | Kurang baik – banyak deviasi               |


- **Qualitative Metrics**

  Evaluasi manual oleh reviewer manusia sangat penting, terutama dalam konteks instruksi prosedural yang membutuhkan keakuratan langkah dan pemahaman domain. Oleh karena itu, disarankan melibatkan domain expert dalam proses review.
  
  Kriteria Penilaian Manual:
  - Apakah langkah-langkah lengkap dan sesuai konteks?
  - Apakah urutan langkah logis dan tidak membingungkan?
  - Apakah ada informasi yang keliru atau tidak relevan?
  - Apakah bahasa instruksi jelas dan mudah dipahami?

  Dapat menggunakan skala 1–5 untuk menilai: Clarity, Correctness, Completeness, dan Relevance.

  `Catatan`: Meskipun metrik otomatis (ROUGE, METEOR) dapat memberikan gambaran awal performa model, penilaian manual tetap krusial karena tidak semua kesalahan bisa dideteksi dengan metrik otomatis, serta beberapa instruksi bisa memiliki multiple correct formats yang secara semantik benar tapi secara tekstual berbeda dan validasi akhir memerlukan penilaian dari pihak yang memahami domain.

- **Benchmarking**

  **Tujuan**: Mengukur peningkatan performa model setelah fine-tuning dibandingkan dengan baseline (base LLM) dan jawaban referensi (human-generated).

  **Komparasi**:
    
    - **Baseline LLM**: Menunjukkan kemampuan model sebelum melihat data spesifik.
    - **Fine-tuned Model**: Model yang telah dilatih dengan instruksi prosedural berbasis produk.
    - **Human Reference**: Sebagai standar kebenaran dalam bentuk instruksi yang benar dan lengkap.

  **Catatan Penting**:
    Benchmarking terhadap baseline tidak sepenuhnya “apple-to-apple” karena baseline belum mengenal konteks produk. Namun tetap berguna sebagai _lower bound_ untuk menilai seberapa jauh fine-tuning meningkatkan pemahaman instruksi spesifik domain.

  **Metode Evaluasi**:
    Menggunakan metrik kuantitatif yang sudah dijelaskan sebelumnya (ROUGE-L, METEOR) dan kualitatif (penilaian domain expert terhadap kejelasan, relevansi, dan kelengkapan).