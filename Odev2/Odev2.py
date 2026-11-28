import csv
import os
from time import perf_counter
from textwrap import fill
import numpy as np
import keras
from keras import layers, initializers, optimizers
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter

# ------------------------------- Temel Ayarlar -------------------------------
SEED = 30
EPOCHS = 1000

GIZLI_KATMAN_SAYISI = 1
GIZLI_KATMAN_NORONLARI = [10]
GIZLI_KATMAN_AKTIVASYONLARI = ["relu"]  # sigmoid, tanh, relu, leaky_relu, elu, prelu, silu
OGRENME_ORANI = 0.01
OPTIMIZER = "adam"                      # "sgd", "adam", "rmsprop"
BASLANGIC_DAGILIMI = "normal"           # "normal" veya "uniform"

GIF_URET = False                        # 1000 epoch GIF uzun sürer; tek seferlik testte True yapabilirsin
CANLI_GRAFIK = False                    # Toplu deneylerde False; tekli incelemede True yapabilirsin
ADIM_ADIM = False
EPOCH_BEKLEME = 0.05
ILK_EPOCH_SAYISI_YAZDIR = 1
YAZDIRMA_ARALIGI = 100
GOSTERILECEK_EPOCHLAR = [0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 750, 1000]

AKTIVASYON_SOZLUGU = {
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "relu": "relu",
    "leakyrelu": "leaky_relu",
    "elu": "elu",
    "prelu": "prelu",
    "silu": "silu",
}


def hedef(p):
    return 1.0 + np.sin(np.pi / 4.0 * p)


def yarim_mse(y_true, y_pred):
    """L = 1/2 * (t - a)^2"""
    return 0.5 * keras.ops.mean(keras.ops.square(y_true - y_pred), axis=-1)


def get_initializer(dagilim, seed=SEED):
    if dagilim == "normal":
        return initializers.RandomNormal(mean=0.0, stddev=0.2, seed=seed)
    if dagilim == "uniform":
        return initializers.RandomUniform(minval=-0.2, maxval=0.2, seed=seed)
    raise ValueError("BASLANGIC_DAGILIMI 'normal' veya 'uniform' olmalıdır.")


def get_optimizer(opt_adi, lr):
    isim = opt_adi.strip().lower()
    if isim == "sgd":
        return optimizers.SGD(learning_rate=lr)
    if isim == "adam":
        return optimizers.Adam(learning_rate=lr)
    if isim == "rmsprop":
        return optimizers.RMSprop(learning_rate=lr)
    raise ValueError("OPTIMIZER 'sgd', 'adam' veya 'rmsprop' olmalıdır.")


def aktivasyon_normalize(aktivasyonlar):
    liste = []
    for i, ad in enumerate(aktivasyonlar):
        anahtar = ad.strip().lower().replace("_", "").replace("-", "") if isinstance(ad, str) else ""
        if anahtar not in AKTIVASYON_SOZLUGU:
            raise ValueError(f"{i + 1}. katman için geçersiz aktivasyon: {ad!r}")
        liste.append(AKTIVASYON_SOZLUGU[anahtar])
    return liste


# ------------------------------- Veri -------------------------------
keras.utils.set_random_seed(SEED)
p_egitim = np.linspace(-2, 2, 81).reshape(-1, 1)
t_egitim = hedef(p_egitim)
p_cizim = np.linspace(-2, 2, 400).reshape(-1, 1)
t_cizim = hedef(p_cizim).ravel()


def build_model(katman_sayisi, noronlar, aktivasyonlar, dagilim="normal", seed=SEED):
    if len(noronlar) != katman_sayisi or len(aktivasyonlar) != katman_sayisi:
        raise ValueError("Katman sayısı ile nöron ve aktivasyon listelerinin uzunluğu eşit olmalıdır.")
    secilen = aktivasyon_normalize(aktivasyonlar)
    k_init = get_initializer(dagilim, seed=seed)
    b_init = initializers.Zeros()

    m = keras.Sequential([keras.Input(shape=(1,))])
    for i, (n, akt) in enumerate(zip(noronlar, secilen), start=1):
        m.add(layers.Dense(n, kernel_initializer=k_init, bias_initializer=b_init, name=f"gizli_{i}"))
        if akt == "prelu":
            m.add(layers.PReLU(name=f"aktivasyon_{i}"))
        elif akt == "leaky_relu":
            m.add(layers.LeakyReLU(negative_slope=0.01, name=f"aktivasyon_{i}"))
        else:
            m.add(layers.Activation(akt, name=f"aktivasyon_{i}"))
    m.add(layers.Dense(1, activation="linear", kernel_initializer=k_init, bias_initializer=b_init, name="cikis"))
    return m, secilen


class EpochKaydedici(keras.callbacks.Callback):
    def __init__(self, yazdir=True):
        super().__init__()
        self.yazdir = yazdir
        self.kayip = []
        self.cikislar = []
        self.agirliklar = []

    def _kaydet_durum(self):
        self.cikislar.append(self.model(p_cizim, training=False).numpy().ravel())
        self.agirliklar.append(np.concatenate([w.ravel() for w in self.model.get_weights()]))

    def _guncel_kayip(self):
        tahmin = self.model(p_egitim, training=False)
        return float(keras.ops.mean(yarim_mse(t_egitim, tahmin)))

    def on_train_begin(self, logs=None):
        ilk = self._guncel_kayip()
        self.kayip.append(ilk)
        self._kaydet_durum()
        if self.yazdir:
            print(f"Epoch    0 | loss = {ilk:.6f}   (başlangıç)")

    def on_epoch_end(self, epoch, logs=None):
        kay = self._guncel_kayip()
        self.kayip.append(kay)
        self._kaydet_durum()
        e = epoch + 1
        if self.yazdir and (e <= ILK_EPOCH_SAYISI_YAZDIR or e % YAZDIRMA_ARALIGI == 0 or e == EPOCHS):
            print(f"Epoch {e:4d} | loss = {kay:.6f}")


def grafik_bilgisi(fig, meta):
    bilgi = (
        f"Deney: {meta['etiket']} | Ağ: {meta['ag_yapisi']} | Seed: {SEED}\n"
        f"Gizli aktivasyonlar: {', '.join(meta['aktivasyonlar'])} | Çıkış: lineer | Başlangıç: {meta['dagilim']}\n"
        f"Optimizer: {meta['opt_adi']} | Öğrenme oranı: {meta['lr']:g} | Epoch: {meta['tamamlanan_epoch']}/{EPOCHS}\n"
        f"Son kayıp: {meta['son_kayip']:.6e} | Eğitim süresi: {meta['sure']:.2f} sn"
    )
    satir_genisligi = max(35, int(fig.get_figwidth() * 10))
    bilgi = "\n".join(fill(satir, width=satir_genisligi) for satir in bilgi.splitlines())
    if not hasattr(fig, "_egitim_bilgisi"):
        fig._egitim_bilgisi = fig.text(
            0.5, 0.98, bilgi, ha="center", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EDF3FA", edgecolor="#7994B5"),
        )
        fig._egitim_bilgisi.set_in_layout(False)
    else:
        fig._egitim_bilgisi.set_text(bilgi)
    bilgi_yuksekligi = 0.25 + 0.19 * len(bilgi.splitlines())
    ust_bosluk = (bilgi_yuksekligi + 0.3) / fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, 1 - ust_bosluk))


def grafikleri_kaydet(kayip, cikislar, meta, cikti_klasoru, gif_uret=False):
    os.makedirs(cikti_klasoru, exist_ok=True)
    tamamlanan = len(kayip) - 1
    epochlar = np.arange(tamamlanan + 1)

    # 1. Loss grafiği
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochlar, kayip, color="#C0392B")
    ax[0].set_title("Her epoch'taki kayıp (doğrusal eksen)")
    ax[1].semilogy(epochlar, np.maximum(kayip, np.finfo(float).tiny), color="#C0392B")
    ax[1].set_title("Her epoch'taki kayıp (logaritmik eksen)")
    for a in ax:
        a.set_xlabel("epoch")
        a.set_ylabel("loss  L = ½(t − a)²")
        a.grid(alpha=0.3)
    grafik_bilgisi(fig, meta)
    plt.savefig(os.path.join(cikti_klasoru, "loss_epoch.png"), dpi=150)
    plt.close(fig)

    # 2. Seçili epoch'larda giriş/çıkış eğrisi
    secili = sorted({0, tamamlanan} | {e for e in GOSTERILECEK_EPOCHLAR if 0 <= e <= tamamlanan})
    sut = 4
    satir = int(np.ceil(len(secili) / sut))
    fig, axs = plt.subplots(satir, sut, figsize=(14, 3.2 * satir), sharex=True, sharey=True, squeeze=False)
    for ax, e in zip(axs.flat, secili):
        ax.plot(p_cizim, t_cizim, color="#1F3A5F", lw=2, label="Hedef")
        ax.plot(p_cizim, cikislar[e], color="#C0392B", lw=2, label="Ağ çıkışı")
        ax.scatter(p_egitim[::4], t_egitim[::4], s=10, color="#1F3A5F", alpha=0.6, zorder=3)
        ax.set_title(f"Epoch {e}   (loss = {kayip[e]:.4f})", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axs.flat[len(secili):]:
        ax.axis("off")
    axs.flat[0].legend(loc="upper left", fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("p (giriş)")
    for ax in axs[:, 0]:
        ax.set_ylabel("çıkış")
    grafik_bilgisi(fig, meta)
    plt.savefig(os.path.join(cikti_klasoru, "giris_cikis_epoch.png"), dpi=150)
    plt.close(fig)

    # 3. Tüm epoch'lar tek grafikte
    fig, ax = plt.subplots(figsize=(8, 5))
    adim = max(1, tamamlanan // 100)
    secilen = sorted(set(range(0, tamamlanan + 1, adim)) | {tamamlanan})
    renkler = cm.viridis(np.linspace(0, 1, len(secilen)))
    for r, e in zip(renkler, secilen):
        ax.plot(p_cizim, cikislar[e], color=r, lw=1, alpha=0.8)
    ax.plot(p_cizim, t_cizim, color="red", lw=2.5, label="Hedef: 1 + sin(πp/4)")
    sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, max(1, tamamlanan)))
    fig.colorbar(sm, ax=ax, label="epoch")
    ax.set_xlabel("p (giriş)")
    ax.set_ylabel("çıkış")
    ax.set_title("Ağ çıkışının epoch'lar boyunca değişimi")
    ax.legend()
    ax.grid(alpha=0.3)
    grafik_bilgisi(fig, meta)
    plt.savefig(os.path.join(cikti_klasoru, "giris_cikis_tum.png"), dpi=150)
    plt.close(fig)

    # 4. İsteğe bağlı GIF
    if gif_uret:
        kareler = list(range(0, tamamlanan + 1, max(1, tamamlanan // 120)))
        if kareler[-1] != tamamlanan:
            kareler.append(tamamlanan)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(p_cizim, t_cizim, color="#1F3A5F", lw=2, label="Hedef")
        cizgi, = ax1.plot([], [], color="#C0392B", lw=2, label="Ağ çıkışı")
        ax1.scatter(p_egitim[::4], t_egitim[::4], s=10, color="#1F3A5F", alpha=0.6)
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-0.2, 2.3)
        ax1.set_xlabel("p (giriş)")
        ax1.set_ylabel("çıkış")
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.3)
        baslik = ax1.set_title("")
        ax2.semilogy(epochlar, np.maximum(kayip, np.finfo(float).tiny), color="#CCCCCC")
        nokta, = ax2.plot([], [], "o", color="#C0392B")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("loss")
        ax2.grid(alpha=0.3)
        grafik_bilgisi(fig, meta)

        def guncelle(e):
            cizgi.set_data(p_cizim.ravel(), cikislar[e])
            nokta.set_data([e], [kayip[e]])
            baslik.set_text(f"Epoch {e}   loss = {kayip[e]:.5f}")
            return cizgi, nokta, baslik

        anim = FuncAnimation(fig, guncelle, frames=kareler, blit=False)
        anim.save(os.path.join(cikti_klasoru, "egitim.gif"), writer=PillowWriter(fps=12))
        plt.close(fig)


def bir_deney_calistir(grup, etiket, katman_sayisi, noronlar, aktivasyonlar,
                        lr, opt_adi, dagilim, epochs=EPOCHS, grafik_kaydet=False,
                        gif_uret=False, yazdir=False):
    keras.utils.set_random_seed(SEED)
    model, secilen = build_model(katman_sayisi, noronlar, aktivasyonlar, dagilim=dagilim, seed=SEED)
    opt = get_optimizer(opt_adi, lr)
    model.compile(optimizer=opt, loss=yarim_mse)

    kayit = EpochKaydedici(yazdir=yazdir)
    t0 = perf_counter()
    model.fit(
        p_egitim, t_egitim,
        epochs=epochs,
        batch_size=len(p_egitim),
        shuffle=False,
        callbacks=[kayit],
        verbose=0,
    )
    sure = perf_counter() - t0

    kayip = np.array(kayit.kayip)
    cikislar = np.array(kayit.cikislar)
    tamamlanan = len(kayip) - 1
    son_kayip = float(kayip[-1])
    en_iyi_kayip = float(np.min(kayip))

    # Tüm eğitim verisi üzerinde son tahmin ve RMSE
    tahmin_son = model(p_egitim, training=False).numpy()
    rmse = float(np.sqrt(np.mean((t_egitim - tahmin_son) ** 2)))

    meta = {
        "etiket": etiket,
        "ag_yapisi": "-".join(map(str, [1, *noronlar, 1])),
        "aktivasyonlar": secilen,
        "dagilim": dagilim,
        "opt_adi": type(opt).__name__,
        "lr": lr,
        "tamamlanan_epoch": tamamlanan,
        "son_kayip": son_kayip,
        "sure": sure,
    }

    if grafik_kaydet:
        klasor = os.path.join("cikti_odev", f"{grup}_{etiket}".replace(" ", "_"))
        grafikleri_kaydet(kayip, cikislar, meta, klasor, gif_uret=gif_uret)

    return {
        "Grup": grup,
        "Deney": etiket,
        "Aktivasyon": ",".join(secilen),
        "LR": lr,
        "Optimizer": type(opt).__name__,
        "Baslangic": dagilim,
        "Epoch": tamamlanan,
        "Son_Loss": round(son_kayip, 6),
        "Min_Loss": round(en_iyi_kayip, 6),
        "RMSE": round(rmse, 6),
        "Sure_sn": round(sure, 3),
    }


def tum_karsilastirmalari_calistir():
    sonuclar = []

    # 1. Aktivasyon fonksiyonları (1 gizli katman, 10 nöron, lr=0.01, adam, normal)
    for akt in ["sigmoid", "tanh", "relu", "leaky_relu", "elu", "prelu", "silu"]:
        sonuclar.append(bir_deney_calistir(
            grup="1_Aktivasyon",
            etiket=akt,
            katman_sayisi=1,
            noronlar=[10],
            aktivasyonlar=[akt],
            lr=0.01,
            opt_adi="adam",
            dagilim="normal",
            epochs=EPOCHS,
            grafik_kaydet=True,
            gif_uret=False,
            yazdir=True,
        ))

    # 2. Öğrenme oranları (relu, 1x10, adam, normal)
    for lr in [0.001, 0.01, 0.1]:
        sonuclar.append(bir_deney_calistir(
            grup="2_OgrenmeOrani",
            etiket=f"lr_{lr}",
            katman_sayisi=1,
            noronlar=[10],
            aktivasyonlar=["relu"],
            lr=lr,
            opt_adi="adam",
            dagilim="normal",
            epochs=EPOCHS,
            grafik_kaydet=True,
            gif_uret=False,
            yazdir=True,
        ))

    # 3. Optimizasyon algoritmaları (relu, 1x10, lr=0.01, normal)
    for opt_adi in ["sgd", "adam", "rmsprop"]:
        sonuclar.append(bir_deney_calistir(
            grup="3_Optimizer",
            etiket=opt_adi,
            katman_sayisi=1,
            noronlar=[10],
            aktivasyonlar=["relu"],
            lr=0.01,
            opt_adi=opt_adi,
            dagilim="normal",
            epochs=EPOCHS,
            grafik_kaydet=True,
            gif_uret=False,
            yazdir=True,
        ))

    # 4. Başlangıç dağılımları (relu, 1x10, lr=0.01, adam -> normal vs uniform)
    for dagilim in ["normal", "uniform"]:
        sonuclar.append(bir_deney_calistir(
            grup="4_BaslangicDagilimi",
            etiket=dagilim,
            katman_sayisi=1,
            noronlar=[10],
            aktivasyonlar=["relu"],
            lr=0.01,
            opt_adi="adam",
            dagilim=dagilim,
            epochs=EPOCHS,
            grafik_kaydet=True,
            gif_uret=False,
            yazdir=True,
        ))

    os.makedirs("cikti_odev", exist_ok=True)
    csv_yolu = os.path.join("cikti_odev", "deney_ozeti.csv")
    with open(csv_yolu, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sonuclar[0].keys()))
        writer.writeheader()
        writer.writerows(sonuclar)

    print("\n=== ÖZET TABLO ===")
    for satir in sonuclar:
        print(satir)
    print(f"\nÖzet CSV kaydedildi: {csv_yolu}")


if __name__ == "__main__":
    tum_karsilastirmalari_calistir()