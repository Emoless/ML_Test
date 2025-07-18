import requests
from bs4 import BeautifulSoup
import re


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Убираем лишние пробелы
    text = re.sub(r'[^\w\s]', '', text)  # Убираем знаки препинания
    return text.strip()

def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Бери текст из основных тегов
        text = ' '.join([tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2', 'div'])])
        return text
    except Exception as e:
        print(f"Ошибка при парсинге {url}: {e}")
        return ""

# Пример: список твоих ссылок
urls = [
  "https://www.factorybuys.com.au/products/euro-top-mattress-king",
  "https://dunlin.com.au/products/beadlight-cirrus",
  "https://themodern.net.au/products/hamar-plant-stand-ash",
  "https://furniturefetish.com.au/products/oslo-office-chair-white",
  "https://hemisphereliving.com.au/products/",
  "https://home-buy.com.au/products/bridger-pendant-larger-lamp-metal-brass",
  "https://interiorsonline.com.au/products/interiors-online-gift-card",
  "https://beckurbanfurniture.com.au/products/page/2/",
  "https://livingedge.com.au/products/tables/dining",
  "https://edenliving.online/collections/summerloving/products/nice-lounge-1",
  "https://www.ourfurniturewarehouse.com.au/products/athens-3pce-lounge-includes-2x-armless-3-seater-and-corner-ottoman-in-grey-storm-fabric?_pos=1&_sid=9f9ca4320&_ss=r",
  "https://cane-line.co.uk/collections/news-2020/products/breeze-sunbed-5569",
  "https://haute-living.com/products/beam-desk",
  "https://www.knoll.com/design-plan/products/by-designer/knoll",
  "https://www.balirepublic.com.au/products/fabric-cleaner",
  "https://vastinterior.com.au/products/samson-daybed-single-2",
  "https://www.hudsonfurniture.com.au/products/string-weave-timber-stool",
  "https://premiumpatio.com.au/products/outdoor-tables/page/3/",
  "https://dhfonline.com/products/gift-card",
  "https://furnish123watertown.com/products/",
  "https://www.tandemarbor.com/products/kaiser-box-bed-blush-plush-velvet",
  "https://www.perchfurniture.com/products/hoyt-chair",
  "http://www.vawayside.net/store/products/tag/beds",
  "https://curiousgrace.com.au/products/cleo-desk-lamp",
  "https://modshop1.com/collections/amalfi/products/amalfi-3-door-credenza",
  "https://www.scandesign.com/products/pavia-sectional-ivory",
  "https://www.sofamania.com/products/carl-victorian-inspired-velvet-button-detail-vanity-accent-stool",
  "https://www.fentonandfenton.com.au/products/gift-card&media=http://cdn.shopify.com/s/files/1/1909/9637/products/GiftVouchers_Product_1024x1024.jpg&description=Gift%20Card%20%23Art-Lover%20%23Birthday%20%23Colour-Lover",
  "https://4-chairs.com/products/mason-chair",
  "https://www.theinside.com/products/x-bench-onyx-austin-stripe-by-old-world-weavers/PTM_XBench_OnyxAustinStripeByOldWorldWeavers",
  "https://stationfurniture.store/products/milano",
  "https://pinchdesign.com/products/yves-desk",
  "https://www.do-shop.com/products/gift-card",
  "https://claytongrayhome.com/products/palecek-coco-ruffle-mirror",
  "https://dfohome.com/products/patio-furniture/swings/all-swings",
  "https://acmeshelving.com/collections/signs/products/sale",
  "https://www.kmpfurniture.com/fire_collection/products/beds_80.html",
  "https://www.jseitz.com/products/buddha",
  "https://www.furnitureworldgalleries.com/products/",
  "https://www.homekoncepts.com/products/furniture/tables/end-tables/",
  "https://cityfurnitureshop.com/collections/greenington/products/azara-bed",
  "https://www.theguestroomfurniture.com/products/",
  "https://www.danishinspirations.com/products/products/bedroom/page/3/",
  "https://columbineshowroom.com/products/",
  "https://emfurn.com/products/aaron-sofa",
  "https://fmvegas.com/products/carlo-dresser",
  "https://nicoyafurniture.com.au/products/playa-bowl",
  "https://galvinbrothers.co.uk/products/alfandbud",
  "https://tyfinefurniture.com/products/enso-platform-bed",
  "https://24estyle.com/products/eliza-fan",
  "https://shopspencerfurnituresiouxfalls.com/products/glass-lamp",
  "https://bryantscountrystore.com/products/deli-sandwich-carryout",
  "https://vincentdesign.com.au/products/spaltekanden-big-white",
  "https://magnolialane.biz/products/anika-cushion",
  "https://blupeter.com.au/products/tulip-object",
  "https://www.cranmorehome.com.au/products/gift-card",
  "https://www.vavoom.com.au/products/ebony-chest",
  "https://www.goodwoodfurniture.com.au/products/wardrobes/colonial-robe/",
  "https://valentinesfurniture.com.au/products/",
  "https://www.loungesplus.com.au/products/baird-fabric-lounge-package",
  "https://www.yellowleafhammocks.com/products/hammock-gift-card",
  "https://asplundstore.se/products/fish-bricka",
  "https://distinctive-interiors.com/products/",
  "https://runyonsfinefurniture.com/products/alf-bar-stool",
  "https://www.comfortfurniture.com.sg/sale/products/office",
  "https://karladubois.com/products/wooster-convertible-crib",
  "https://brownsquirrelfurniture.com/furniture/rest-bedrooms/rest-beds/sleep-queen/products/laurel-sleigh/",
  "https://viesso.com/products/haru-bed",
  "https://floydhome.com/products/the-floyd-hat",
  "https://www.mybudgetfurniture.com/products/3pc-sectional",
  "https://www.fiveelementsfurniture.com/collections/sola-office-collection/products/sola-lift-desk",
  "https://www.fads.co.uk/products/living/sofas/sofa-beds/",
  "https://test-danish-inspirations.pantheonsite.io/products/products/bedroom/page/3/",
  "https://www.hipvan.com/products/sleep-mattress?ref=nav_dropdown",
  "https://dev-danish-inspirations.pantheonsite.io/products/products/bedroom/page/3/",
  "https://houseofhollingsworthblog.com/products/locker",
  "https://www.wardrobe-bunk-bed-sofa.uk/products/wardrobe-ava-4-3",
  "https://dwell.co.uk/products/sleep/type/bed/",
  "https://www.hopewells.co.uk/products/living-dining/sofas-chairs/tetrad/tetrad-patna-chair",
  "https://kokocollective.com.au/products/esme-natural-rattan-day-bed",
  "https://www.timothyoulton.com/products/living/sectional-sofas",
  "https://www.warmnordic.com/global/products/news",
  "https://www.royaloakfurniture.co.uk/products/pop-bench",
  "https://mulamu.com/products/membership",
  "http://www.genesisfurniturelv.com/products/",
  "https://daufurniture.com/products/style/transitional/dana-upholstered-bed/",
  "https://www.waynesfinefurnitureandbedding.com/search/products/?s=power+recliners&pp=120",
  "https://theodores.com/products/",
  "https://www.theoakfurnitureshop.com/products/country-trend-solid-oak-tv-stand-51/",
  "https://totalpatioaccessories.com/products/product07",
  "https://www.groensfinefurniture.com/products/Miscellaneous/misc/394554.html",
  "https://yesterdaystreefurniture.com/products/gifts/",
  "https://www.jensen-lewis.com/products/guardsman-gold-complete-plan",
  "https://www.popandscott.com/products/raffia-sun",
  "https://shophorne.com/products/sarpaneva-cast-iron-casserole",
  "https://cultdesign.co.nz/products/molloy-chair",
  "https://www.gloster.com/en/products/materials/teak",
  "https://thebrick.com/products/tacoma-chest"
]  # Замени на свои 700 ссылок
for url in urls:
    text = clean_text(get_text_from_url(url))
    with open("raw_data.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")