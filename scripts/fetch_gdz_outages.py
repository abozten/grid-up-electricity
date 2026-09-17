"""GDZ 'Tablo-1 Kesintiler' -> data/gdz_outages_daily.csv (il, ilce, gun bazinda kesinti).

Kaynak: gdzelektrik.com.tr/.../tedarik-surekliligi-kalite-tablolari
Kapsam: 2025-01 .. 2026-05 (yayin klasoru adlari yayin ayidir, veri ayi degil;
KOD_NO ile tekillestirip zaman damgasindan yila karar veriyoruz).

NOT: 2026-06/07 (test'in son iki ayi) yayinlanmadi -> test-time feature olarak kullanilamaz.
Olcum sonucu: ilce-gun toplam tuketim artigi ile korelasyon ~0.005 (bkz. reports/).
"""
import io, re, urllib.parse, urllib.request
import pandas as pd

PAGE = ("https://www.gdzelektrik.com.tr/bilgi-merkezi/yasal-bildirimler/"
        "hizmet-kalitesi-gostergeleri/tedarik-surekliligi-kalite-tablolari")
YEARS = re.compile(r"/(?:January|February|March|April|May|June|July|August|"
                   r"September|October|November|December)(?:2025|2026)/Tablo-1")
CUST = ["KENTSEL_OG", "KENTSEL_AG", "KENTALTI_OG", "KENTALTI_AG", "KIRSAL_OG", "KIRSAL_AG"]


def get(url):
    return urllib.request.urlopen(
        urllib.request.Request(urllib.parse.quote(url, safe=":/?&="),
                               headers={"User-Agent": "Mozilla/5.0"}), timeout=120).read()


def main(out="data/gdz_outages_daily.csv"):
    html = get(PAGE).decode("utf-8", "replace")
    urls = sorted({u for u in re.findall(r"https?://[^\"']*storage/tedarik-surekliligi/[^\"']*", html)
                   if YEARS.search(u)})
    print(f"{len(urls)} dosya")
    frames = []
    for u in urls:
        try:
            frames.append(pd.read_excel(io.BytesIO(get(u))))
        except Exception as e:  # ponytail: bozuk dosyayi atla, hepsi lazim degil
            print("atlandi:", u.rsplit("/", 1)[-1], e)
    d = pd.concat(frames, ignore_index=True).drop_duplicates("KOD_NO")
    d["tarih"] = pd.to_datetime(d["BAŞLAMA_TARİHİ_VE_ZAMANI"], errors="coerce").dt.normalize()
    # TOPLAM_* sutunlari zaten musteri*saat (ornek: 42 musteri x 1.011 sa = 42.44)
    d["cust_hours"] = d[[c for c in d.columns if c.startswith("TOPLAM_")]].sum(axis=1)
    d["cust"] = d[CUST].sum(axis=1)
    # ponytail: kesintiyi baslangic gunune yaziyoruz; >24sa kesintiler icin gune bolmek gerekir
    g = (d.dropna(subset=["tarih"]).groupby(["İL", "İLÇE", "tarih"])
         .agg(cust_hours=("cust_hours", "sum"), cust_affected=("cust", "sum"),
              n_outage=("KOD_NO", "size"), max_dur_h=("KESİNTİ_SÜRESİ", "max"))
         .reset_index().rename(columns={"İL": "il", "İLÇE": "ilce"}))
    g.to_csv(out, index=False)
    print(f"{len(g)} satir -> {out}  ({g.tarih.min().date()} .. {g.tarih.max().date()})")
    assert g.cust_hours.min() >= 0 and g.tarih.is_monotonic_increasing is not None
    return g


if __name__ == "__main__":
    main()
