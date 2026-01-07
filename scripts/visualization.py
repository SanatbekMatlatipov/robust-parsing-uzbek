import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import io

# ---------------------------------------------------------
# 1. LOAD THE XML DATA
# ---------------------------------------------------------
xml_data = """<?xml version="1.0" encoding="UTF-8"?>
<treebank>
  <size>
    <total><sentences>681</sentences><tokens>7542</tokens><words>7542</words><fused>0</fused></total>
    <train><sentences>483</sentences><tokens>5441</tokens><words>5441</words><fused>0</fused></train>
    <dev><sentences>0</sentences><tokens>0</tokens><words>0</words><fused>0</fused></dev>
    <test><sentences>198</sentences><tokens>2101</tokens><words>2101</words><fused>0</fused></test>
  </size>
  <tags unique="16">
    <tag name="ADJ">517</tag>
    <tag name="ADP">118</tag>
    <tag name="ADV">204</tag>
    <tag name="AUX">87</tag>
    <tag name="CCONJ">86</tag>
    <tag name="DET">116</tag>
    <tag name="INTJ">23</tag>
    <tag name="NOUN">2508</tag>
    <tag name="NUM">180</tag>
    <tag name="PART">43</tag>
    <tag name="PRON">473</tag>
    <tag name="PROPN">26</tag>
    <tag name="PUNCT">1567</tag>
    <tag name="SCONJ">9</tag>
    <tag name="VERB">1578</tag>
    <tag name="X">7</tag>
  </tags>
  <feats unique="54">
    <feat name="Aspect" value="Hab" upos="VERB">62</feat>
    <feat name="Aspect" value="Perf" upos="VERB">256</feat>
    <feat name="Case" value="Abl" upos="ADJ,NOUN,PRON,VERB">94</feat>
    <feat name="Case" value="Acc" upos="ADJ,NOUN,NUM,PRON,PROPN,VERB">262</feat>
    <feat name="Case" value="Dat" upos="ADJ,ADV,NOUN,PRON,PROPN,VERB">210</feat>
    <feat name="Case" value="Gen" upos="ADJ,NOUN,PRON,PROPN">158</feat>
    <feat name="Case" value="Loc" upos="ADJ,NOUN,NUM,PRON,PROPN,VERB">143</feat>
    <feat name="Case" value="Nom" upos="NOUN,NUM,PRON,PROPN,VERB">1537</feat>
    <feat name="Mood" value="Ind" upos="AUX,VERB">302</feat>
    <feat name="Number" value="Plur" upos="ADJ,NOUN,PRON,VERB">384</feat>
    <feat name="Number" value="Sing" upos="AUX,NOUN,PRON,PROPN,VERB">523</feat>
    <feat name="Number[psor]" value="Plur,Sing" upos="ADJ,NOUN,NUM,PRON,VERB">353</feat>
    <feat name="Person" value="3" upos="AUX,PRON,VERB">412</feat>
    <feat name="Person[psor]" value="3" upos="ADJ,NOUN,NUM,PRON,VERB">433</feat>
    <feat name="PronType" value="Dem" upos="ADV,DET,PRON">123</feat>
    <feat name="PronType" value="Prs" upos="PRON">239</feat>
    <feat name="Tense" value="Past" upos="AUX,VERB">361</feat>
    <feat name="VerbForm" value="Conv" upos="VERB">324</feat>
    <feat name="VerbForm" value="Fin" upos="AUX,VERB">382</feat>
  </feats>
  <deps unique="38">
    <dep name="acl">138</dep>
    <dep name="advcl">441</dep>
    <dep name="advmod">210</dep>
    <dep name="amod">445</dep>
    <dep name="compound">265</dep>
    <dep name="conj">199</dep>
    <dep name="det">184</dep>
    <dep name="nmod">235</dep>
    <dep name="nmod:poss">189</dep>
    <dep name="nsubj">717</dep>
    <dep name="nummod">131</dep>
    <dep name="obj">416</dep>
    <dep name="obl">732</dep>
    <dep name="punct">1567</dep>
    <dep name="root">681</dep>
    <dep name="xcomp">139</dep>
  </deps>
</treebank>
"""

root = ET.fromstring(xml_data)

# ---------------------------------------------------------
# 2. GENERATE TABLE DATA (Splits)
# ---------------------------------------------------------
print("--- TABLE 1: DATASET STATISTICS ---")
print(f"{'Split':<10} | {'Sentences':<10} | {'Tokens':<10}")
print("-" * 36)

splits = ['train', 'test', 'total']
for split in splits:
    node = root.find(f"./size/{split}")
    sents = node.find('sentences').text
    tokens = node.find('tokens').text
    print(f"{split.capitalize():<10} | {sents:<10} | {tokens:<10}")
print("\n")


# ---------------------------------------------------------
# 3. GENERATE FIGURE 1: UPOS DISTRIBUTION
# ---------------------------------------------------------
tags = {}
for tag in root.findall("./tags/tag"):
    tags[tag.get('name')] = int(tag.text)

# Sort and prepare data
sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
labels = [x[0] for x in sorted_tags]
sizes = [x[1] for x in sorted_tags]

# Group small values into "Other" for cleaner chart
if len(labels) > 9:
    labels = labels[:9] + ['OTHER']
    sizes = sizes[:9] + [sum(sizes[9:])]

plt.figure(figsize=(10, 7))
colors = plt.cm.Paired(range(len(labels)))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, pctdistance=0.85, textprops={'fontsize': 10})
centre_circle = plt.Circle((0,0),0.70,fc='white')
plt.gcf().gca().add_artist(centre_circle)
plt.title('Distribution of Universal POS Tags', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Figure1_UPOS_Distribution.png', dpi=300)
print("Generated Figure1_UPOS_Distribution.png")
plt.show()


# ---------------------------------------------------------
# 4. GENERATE FIGURE 2: DEPENDENCY RELATIONS
# ---------------------------------------------------------
deps = {}
for dep in root.findall("./deps/dep"):
    deps[dep.get('name')] = int(dep.text)

# Get Top 20 Relations
sorted_deps = sorted(deps.items(), key=lambda x: x[1], reverse=True)[:20]
labels = [x[0] for x in sorted_deps]
values = [x[1] for x in sorted_deps]

plt.figure(figsize=(12, 6))
bars = plt.bar(labels, values, color='#4C72B0', zorder=3)
plt.title('Top 20 Dependency Relations', fontsize=14, fontweight='bold')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# Add counts on top
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5, f'{int(height)}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('Figure2_DepRel_Frequency.png', dpi=300)
print("Generated Figure2_DepRel_Frequency.png")
plt.show()


# ---------------------------------------------------------
# 5. GENERATE FIGURE 3: TOP MORPHOLOGICAL FEATURES
# ---------------------------------------------------------
feats = {}
for feat in root.findall("./feats/feat"):
    # Combine Feature Name and Value (e.g., "Case=Nom")
    name = f"{feat.get('name')}={feat.get('value')}"
    feats[name] = int(feat.text)

# Get Top 15 Features
sorted_feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)[:15]
labels = [x[0] for x in sorted_feats]
values = [x[1] for x in sorted_feats]

plt.figure(figsize=(10, 8))
y_pos = range(len(labels))
plt.barh(y_pos, values, color='#55A868', zorder=3)
plt.yticks(y_pos, labels)
plt.gca().invert_yaxis() # Highest on top
plt.title('Top 15 Morphological Features', fontsize=14, fontweight='bold')
plt.xlabel('Frequency')
plt.grid(axis='x', linestyle='--', alpha=0.7, zorder=0)

plt.tight_layout()
plt.savefig('Figure3_Morph_Features.png', dpi=300)
print("Generated Figure3_Morph_Features.png")
plt.show()
