# Eğitim verisi: [Girdi1, Girdi2, Hedef_Etiket]
veri_seti = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
]

# Elle rastgele belirlenen başlangıç ağırlıkları ve bias
w1 = 0.3
w2 = -0.5
b = 0.2
ogrenme_orani = 0.1

epoch = 0

# Denklemin toplam hatası (loss/error) 0 olana kadar döner
while True:
    epoch += 1
    toplam_hata = 0
    
    # Batch size = Epoch size (tüm veri seti her turda toplu işlenir)
    for x1, x2, hedef in veri_seti:
        # Net girdi denklemi: z = w1*x1 + w2*x2 + b
        net_girdi = (x1 * w1) + (x2 * w2) + b
        
        # Basamak aktivasyon fonksiyonu (Step function: 0 veya 1)
        tahmin = 1 if net_girdi >= 0 else 0
        
        # Hata hesaplama
        hata = hedef - tahmin
        toplam_hata += abs(hata)
        
        # Ağırlık güncelleme (Perceptron kuralı)
        w1 += ogrenme_orani * hata * x1
        w2 += ogrenme_orani * hata * x2
        b += ogrenme_orani * hata
        
    print(f"Epoch {epoch} tamamlandı -> Toplam Hata: {toplam_hata}")
    
    if toplam_hata == 0:
        print(f"\nModel egitildi! {epoch}. epoch sonunda hata 0 oldu.")
        print(f"Son Agirliklar: w1={round(w1, 2)}, w2={round(w2, 2)}, bias={round(b, 2)}")
        break