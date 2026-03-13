"""
Uzbek stop words list.

A curated set of common Uzbek function words, particles, and pronouns
that typically carry little semantic content — used by spaCy's built-in
IS_STOP token attribute.
"""

STOP_WORDS = set(
    """
    va ham yoki lekin ammo bilan uchun haqida to'g'risida
    bu shu o'sha u ular biz siz men sen
    edi emas bo'ldi bo'lgan bo'lsa kerak mumkin lozim
    bir iki har hamma barcha ba'zi ayrim ko'p oz kop
    agar gar garchi chunki sababli holda negaki ya'ni demak binobarin
    kabi singari o'xshash yanglig' misol masalan xususan ayniqsa
    yo'q ha yo ha'a albatta shubhasiz
    ning ga da dan ni dir lar ekan ekanligini
    bo'lib hisoblanadi deb degani degan
    juda eng nihoyatda ancha biroz ozgina sal
    hali hamon hech qachon doimo hamisha bazan ko'pincha kamdan-kam
    yuqori pastda oldinda orqada yonida ichida tashqarida o'rtasida
    shu bois natijada oqibatda pirovardida
    """.split()
)
