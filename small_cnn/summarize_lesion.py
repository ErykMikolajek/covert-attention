"""Render scientific figures and Polish report from completed lesion results."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT, TASKS
from lesion import PRIMARY

R=ROOT/'lesion_results'
r=json.loads((R/'results.json').read_text())
a=json.loads((R/'attention_results.json').read_text())
cross=json.loads((R/'cross_replacement_checks.json').read_text())
readout=json.loads((R/'readout_controls.json').read_text())
single=list(csv.DictReader((R/'single_unit_lesions.csv').open()))
active=r['active_units']
names=['Posner','Visual search','Contextual cueing']


def group(task,mode='mean',k=8):
    t=TASKS.index(task)
    return next(g for g in r['groups'] if g['target']==task and g['mode']==mode and g['k']==k and g['strategy']=='selective' and g['noise']==PRIMARY[t])


fig,axes=plt.subplots(2,2,figsize=(12,11),gridspec_kw={'width_ratios':[1,1.5]})
for row,mode in enumerate(['mean','zero']):
    mat=np.array([[100*float(next(s for s in single if s['mode']==mode and int(s['unit'])==u and float(s['noise'])==PRIMARY[t])[f'drop_{task}']) for t,task in enumerate(TASKS)] for u in active])
    ax=axes[row,0];im=ax.imshow(mat,cmap='RdBu_r',vmin=-20,vmax=20,aspect='auto')
    ax.set_xticks(range(3),names,rotation=15,ha='right');ax.set_yticks(range(len(active)),active)
    ax.set_ylabel('Dense unit ID (zero-based)');ax.set_title(f'Single units: {mode} replacement')
    fig.colorbar(im,ax=ax,label='Accuracy drop (percentage points)',shrink=.75)
    gm=np.array([group(t,mode)['accuracy_drop'] for t in TASKS])*100
    ax=axes[row,1];im=ax.imshow(gm,cmap='RdBu_r',vmin=-35,vmax=35)
    ax.set_xticks(range(3),names,rotation=15,ha='right')
    ax.set_yticks(range(3),[n+' group' for n in names]);ax.set_title(f'Groups of 8 selected on validation: {mode}')
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{gm[i,j]:+.1f}',ha='center',va='center',color='white' if abs(gm[i,j])>20 else 'black')
    fig.colorbar(im,ax=ax,label='Accuracy drop (percentage points)',shrink=.75)
fig.suptitle('Dense lesions: Posner/search at SD .08; context at SD .14\nEach group row is evaluated at its target-task noise level',fontsize=11)
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(R/'task_lesions.png',dpi=160);plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(13,3.8))
for t,ax in enumerate(axs):
    gs=[group(TASKS[t],k=k) for k in [1,4,8,16]]
    own=np.array([g['accuracy_drop'][t] for g in gs])*100
    random=np.array([g['random_drop_mean'][t] for g in gs])*100
    lo=np.array([g['random_drop_ci95'][t][0] for g in gs])*100
    hi=np.array([g['random_drop_ci95'][t][1] for g in gs])*100
    ax.plot([1,4,8,16],own,'o-',label='Validation-selected group')
    ax.plot([1,4,8,16],random,'o--',label='Random active units')
    ax.fill_between([1,4,8,16],lo,hi,alpha=.18,label='95% random-group range')
    ax.set(title=names[t],xlabel='Number of units replaced by training mean',ylabel='Target accuracy drop (pp)')
    ax.grid(alpha=.2)
axs[0].legend(fontsize=7)
fig.tight_layout();fig.savefig(R/'lesion_dose.png',dpi=170);plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(13,3.8))
for t,g in enumerate(a['primary']):
    ax=axs[t]
    ax.bar(['Baseline','Lesion'],[g['baseline_effect']*100,g['lesioned_effect']*100],color=['#2878a0','#d7782b'])
    ax.set(title=names[t],ylabel='Attention effect (percentage points)')
    ax.axhline(0,color='gray',lw=.7)
    ax.text(.5,.96,f"Target accuracy drop: {g['accuracy_drop']*100:+.2f} pp",transform=ax.transAxes,ha='center',va='top',fontsize=9)
fig.suptitle('Secondary analysis: groups selected for effect reduction with an accuracy penalty\nMean replacement, 8 units; Posner effect shown before criterion control',fontsize=10)
fig.tight_layout(rect=(0,0,1,.88));fig.savefig(R/'attention_lesions.png',dpi=170);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12,4.2))
x=np.arange(3);fixed=[];calibrated=[];within=[];aucm=[]
for t,g in enumerate(readout):
    primary=next(p for p in r['primary'] if p['target']==g['target'])
    tc=g['threshold_checks'][t]
    fixed.append(100*primary['accuracy_drop'][t]);calibrated.append(100*tc['calibrated_accuracy_drop'])
    within.append(100*g['within_task_mean_accuracy_drop'][t])
    aucm.append([s['baseline_auroc']-s['lesioned_auroc'] for s in g['threshold_checks']])
axs[0].bar(x-.23,fixed,.23,label='Fixed readout, global mean')
axs[0].bar(x,calibrated,.23,label='Validation-recalibrated threshold')
axs[0].bar(x+.23,within,.23,label='Within-task training mean')
axs[0].set_xticks(x,names,rotation=10);axs[0].set_ylabel('Target accuracy drop (pp)');axs[0].legend(fontsize=7)
axs[0].set_title('Same frozen unit IDs: readout controls')
im=axs[1].imshow(aucm,vmin=-.15,vmax=.15,cmap='RdBu_r')
axs[1].set_xticks(x,names,rotation=10);axs[1].set_yticks(x,[n+' group' for n in names])
for i in range(3):
    for j in range(3):axs[1].text(j,i,f'{aucm[i][j]:.3f}',ha='center',va='center',color='white' if abs(aucm[i][j])>.08 else 'black')
axs[1].set_title('Threshold-independent AUROC drop')
fig.colorbar(im,ax=axs[1],shrink=.8)
fig.tight_layout();fig.savefig(R/'readout_controls.png',dpi=170);plt.close(fig)

probes=list(csv.DictReader((R/'posner_tuning_probes.csv').open()))
probe_mat=[]
for u in active:
    left=next(x for x in probes if int(x['unit'])==u and x['side']=='left')
    right=next(x for x in probes if int(x['unit'])==u and x['side']=='right')
    probe_mat.append([float(left['target_auc_cue_0']),float(right['target_auc_cue_0']),float(left['cue_only_auc']),float(right['cue_only_auc'])])
fig,ax=plt.subplots(figsize=(6,8));im=ax.imshow(probe_mat,vmin=0,vmax=1,cmap='RdBu_r',aspect='auto')
ax.set_yticks(range(len(active)),active);ax.set_xticks(range(4),['Target left','Target right','Cue left','Cue right'],rotation=20,ha='right')
ax.set_ylabel('Dense unit ID');ax.set_title('Descriptive single-location probes: AUROC\n512 trials/class, SD .08; no significance classification')
fig.colorbar(im,ax=ax,label='AUROC');fig.tight_layout();fig.savefig(R/'tuning_probes.png',dpi=160);plt.close(fig)

lines=['# Lezje neuronów małej CNN — wyniki eksperymentu','',
       '**Wniosek:** konkretne grupy jednostek wpływają przyczynowo na decyzje tej sieci, lecz duże spadki accuracy częściowo wynikają z przesunięcia kryterium odpowiedzi. Najbardziej stabilną specjalizację informacyjną potwierdzono dla visual search; słabszą dla kontekstu. Pierwotna selektywność grupy Posnera nie utrzymuje się po kontroli progu. Osobna analiza wskazuje częściowy wpływ innej grupy na cueing Posnera. Dla efektów search i contextual cueing główny wariant nie potwierdził selektywnej redukcji na nowych próbach.','',
       '## Model i protokół','',
       f"Analizowany checkpoint: epoka {r['config']['checkpoint_epoch']}, SHA256 `{r['config']['checkpoint_sha256']}`. Wagi pozostały zamrożone. Interwencja dotyczy 64 aktywacji po ReLU w `classifier[2]`, przed liniowym wyjściem. Identyfikatory neuronów są liczone od zera.", '',
       f"Na 30 000 prób kalibracyjnych ze zbioru treningowego zmienną aktywność miały **{len(active)} z 64 jednostek**. Aktywne ID: {active}. Kontrole losowe obejmowały te jednostki; nie losowano głównie nieaktywnych neuronów.", '',
       f"Wybór grup: {r['config']['n_val']} nowych prób na zadanie, seed {r['config']['selection_seed']}. Potwierdzenie: **{r['config']['n_test']} nowych prób na zadanie**, łącznie {3*r['config']['n_test']:,}, seed {r['config']['confirmation_seed']}, przy każdym z SD 0,04 / 0,08 / 0,14. Pary baseline–lezja otrzymują identyczny obraz i szum. Nowe konfiguracje kontekstu są rozłączne z treningiem, wyborem i poprzednim testem. Stare konfiguracje z definicji się powtarzają.", '',
       'Zastępowanie aktywacji: **mean** = globalna średnia z treningu (bez znajomości zadania); **zero** = 0. Kontrole: 200 losowych grup aktywnych jednostek dla każdej wielkości 4, 8 i 16; dla pojedynczych jednostek wszystkie aktywne ID. Jedna próba szumu przypada na każdy czysty obraz w danym warunku; nie traktujemy kolejnych skal szumu jako niezależnych replikacji sieci.', '',
       'Główny wariant ustalony w kodzie przed oceną potwierdzającą: **8 jednostek, mean, strategia selective**. Posner i search oceniane przy SD=0,08, kontekst przy SD=0,14, aby ograniczyć efekt sufitu. Dobór zachłanny na walidacji zwiększa względny przyrost cross-entropy celu i odejmuje średni dodatni przyrost straty pozostałych zadań. Dodatkowo zbadano strategię importance, grupy 1/4/16 i zero; są zapisane w JSON.', '',
       'Główna trafność jest liczona przy znanym zadaniu (dwie odpowiednie klasy). Osobno zapisano wynik sześciu klas i rozpoznawanie rodzaju zadania, aby odróżnić deficyt decyzji od deficytu rozpoznania zadania.', '',
       '## 1. Wpływ grup na rozwiązywanie zadań','',
       'Pierwsza tabela pokazuje interwencję przy oryginalnym, zamrożonym odczycie. Nie należy interpretować całego spadku accuracy jako utraty informacji: niżej znajdują się kontrole progu i AUROC.','',
       '| Grupa wybrana dla | SD | ID neuronów | Accuracy przed → po | Spadek (95% CI), pp | Spadki innych zadań, pp | q względem losowych grup |',
       '|---|---:|---|---:|---:|---|---:|']
for g in r['primary']:
    t=TASKS.index(g['target']);base=r['baseline'][str(g['noise'])]['accuracy'][t];ci=g['paired_accuracy_drop_ci95'][t]
    others=', '.join(f"{names[j]}: {100*g['accuracy_drop'][j]:+.2f}" for j in range(3) if j!=t)
    lines.append(f"| {names[t]} | {g['noise']} | {g['units']} | {base:.2%} → {g['scores']['accuracy'][t]:.2%} | {100*g['accuracy_drop'][t]:.2f} ({100*ci[0]:.2f}–{100*ci[1]:.2f}) | {others} | {g['q_random_selectivity_bh']:.4f} |")
common=sorted(set.intersection(*(set(g['units']) for g in r['primary'])))
lines+=['',f'**Jednostki wspólne dla wszystkich trzech głównych grup: {common}.** Grupy nie są rozłącznymi modułami. Szczególnie grupa kontekstowa wpływa również na pozostałe zadania. Wyniki pokazują względną specjalizację w tej sieci, a nie absolutną wyłączność funkcji.','',
        'Selektywność testowa = spadek accuracy zadania docelowego minus największa bezwzględna zmiana pozostałych zadań. q oznacza korektę Benjamini–Hochberga trzech porównań dla empirycznych ogonów rozkładu 200 losowych grup; minimalna wartość surowego p wynosi 1/201. To porównanie grup wewnątrz jednej sieci, nie test populacji niezależnych treningów.', '',
        'CI: sparowany bootstrap 1500 powtórzeń. W kontekście stare próby są resamplowane na poziomie 16 powtarzanych konfiguracji; nowe osobno. CI warunkowe względem wybranego checkpointu i grup nie obejmuje zmienności treningu ani procedury wyboru grup.','',
        '### Kontrola: te same ID przy zerowaniu','',
        'Poniżej zerujemy dokładnie grupy wybrane dla mean; nie zastępujemy ich innymi, ponownie wybranymi ID.','',
        '| Grupa | Spadek Posner, pp | Spadek search, pp | Spadek context, pp |', '|---|---:|---:|---:|']
for task in TASKS:
    x=next(x for x in cross if x['family']=='task' and x['target']==task and x['selection_mode']=='mean' and x['intervention_mode']=='zero')
    lines.append('| '+task+' | '+' | '.join(f'{d*100:+.2f}' for d in x['accuracy_drop'])+' |')
lines+=['', 'Zmiana definicji interwencji zmienia selektywność tych samych ID. Przykładowo grupa Posnera wybrana dla mean przy zerowaniu silniej pogarsza search niż Posnera. To ogranicza wniosek o stałej, niezależnej od interwencji grupie wyspecjalizowanej w Posnerze.', '',
        '### Kontrola informacji i odczytu: kluczowa dla interpretacji', '',
        'Zamiana aktywacji na globalną średnią może przesunąć logity, powodując częste wybieranie jednej klasy. Zbadano więc: (1) AUROC niezależne od progu, (2) diagnostyczne przekalibrowanie wyłącznie progu decyzji na walidacji — osobno dla baseline i sieci po lezji, (3) zastępowanie aktywacji średnią treningową warunkowaną rodzajem zadania. Ostatnia kontrola zachowuje przeciętną aktywność w danym zadaniu. Są to analizy diagnostyczne po głównym eksperymencie; nie zmieniają oryginalnego checkpointu ani wybranych grup.', '',
        '| Grupa | AUROC celu przed → po | Spadek accuracy po kalibracji, pp | Największy bezwzględny spadek innych zadań po kalibracji, pp | Spadek celu przy średniej dla zadania, pp |',
        '|---|---:|---:|---:|---:|']
for g in readout:
    t=TASKS.index(g['target']);tc=g['threshold_checks'][t]
    other=max(abs(s['calibrated_accuracy_drop']) for j,s in enumerate(g['threshold_checks']) if j!=t)
    lines.append(f"| {g['target']} | {tc['baseline_auroc']:.4f} → {tc['lesioned_auroc']:.4f} | {100*tc['calibrated_accuracy_drop']:.2f} | {100*other:.2f} | {100*g['within_task_mean_accuracy_drop'][t]:.2f} |")
lines+=['', '**Interpretacja:** dla search spadek informacji i trafności pozostaje selektywny po kontroli odczytu. Dla kontekstu pozostaje słabsza względna specjalizacja. Dla grupy Posnera po kalibracji większy deficyt występuje w search niż w Posnerze; pierwotna duża różnica przy stałym progu nie potwierdza zatem selektywnej utraty informacji Posnera. Grupa kontekstowa przy średniej warunkowej nadal może silnie przesuwać kryterium: sam ten wariant również nie zastępuje kontroli AUROC i kalibracji.', '']
lines+=['','### Najsilniejsze pojedyncze jednostki (mean)','', '| Zadanie | ID: spadek accuracy, pp |','|---|---|']
for t,task in enumerate(TASKS):
    candidates=[s for s in single if s['mode']=='mean' and float(s['noise'])==PRIMARY[t]]
    candidates.sort(key=lambda s:float(s[f'drop_{task}']),reverse=True)
    lines.append('| '+names[t]+' | '+', '.join(f"{s['unit']}: {float(s[f'drop_{task}'])*100:+.2f}" for s in candidates[:3])+' |')
lines+=['','## 2. Grupy odpowiadające za efekty uwagowe','',
        'To osobny dobór na walidacji. Cel = zmniejszenie odpowiedniego efektu minus dwukrotność dodatniego spadku accuracy zadania docelowego; k=8. Kontrole losowe oceniono tą samą funkcją. Grupy nie były optymalizowane do oszczędzania pozostałych dwóch zadań — ich wyniki zapisano w JSON.','',
        'Efekt Posnera = HR valid − HR invalid, tylko dla widocznych ramek (kontrast >0). Efekt search = accuracy SS1 − SS8. Efekt kontekstu = accuracy old − new. Zanik efektu po sprowadzeniu całego zadania do losowości sam w sobie nie dowodzi selektywnej lezji mechanizmu uwagi.','',
        '| Zadanie | ID neuronów (mean) | Efekt przed → po, pp | Spadek accuracy zadania, pp | 95% CI zmniejszenia efektu, pp | q surowego celu względem losowych grup |',
        '|---|---|---:|---:|---:|---:|']
for g in a['primary']:
    ci=g['paired_effect_drop_ci95']
    lines.append(f"| {g['target']} | {g['units']} | {100*g['baseline_effect']:.2f} → {100*g['lesioned_effect']:.2f} | {100*g['accuracy_drop']:+.2f} | {100*ci[0]:+.2f} do {100*ci[1]:+.2f} | {g['q_random_objective_bh']:.4f} |")
pos=next(g for g in a['primary'] if g['target']=='posner');c=pos['criterion_control']
lines+=['','### Posner: oddzielenie zmiany czułości od progu decyzji','',
        f"Surowy hit-rate gap zmniejszył się o {100*pos['effect_drop']:.2f} pp przy spadku accuracy Posnera {100*pos['accuracy_drop']:.2f} pp. Jednocześnie FAR w próbach z widoczną ramką wzrósł z {c['test_baseline_far']:.2%} do {c['test_uncalibrated_lesion_far']:.2%}. Część pozornego usunięcia efektu jest więc związana ze zmianą kryterium odpowiedzi.", '',
        f"Próg dla sieci po lezji wyznaczono wyłącznie na walidacji, dopasowując FAR do baseline. Na nowych próbach FAR wyniósł {c['test_matched_far']:.2%} wobec {c['test_baseline_far']:.2%} baseline. Hit-rate gap po tej kontroli: {100*c['baseline_hit_gap']:.2f} → {100*c['matched_hit_gap']:.2f} pp. Redukcja wynosi **{100*c['matched_hit_gap_drop']:.2f} pp**, opisowe 95% CI {100*c['matched_hit_gap_drop_ci95'][0]:.2f}–{100*c['matched_hit_gap_drop_ci95'][1]:.2f} pp.", '',
        f"Różnica AUROC valid–invalid, niezależna od stałego progu, zmalała z {c['auroc']['baseline']['gap']:.4f} do {c['auroc']['lesion']['gap']:.4f}. To opisowa kontrola; nie wykonano osobnego testu istotności AUROC.", '',
        'Ten wynik daje częściową przesłankę wpływu grupy na cueing w wariancie mean. Nie oznacza izolacji uniwersalnego modułu uwagi: grupa zmienia także inne zadania, a kontrola progu jest analizą diagnostyczną po głównym eksperymencie.','',
        'Dla zerowania grup wybranych pod zero oraz dla tych samych ID wybranych pod mean pełne wyniki kontroli są w `attention_results.json` i `cross_replacement_checks.json`. Wyniku mean nie należy automatycznie przenosić na zero.','',
        '**Visual search i contextual cueing:** główne grupy mean wybrane do zmniejszenia efektów nie potwierdziły takiej funkcji na nowych próbach. Efekty po lezji nawet wzrosły. Warianty zero są wynikami dodatkowymi; same nie stanowią potwierdzenia stabilnej, niezależnej od definicji lezji grupy mechanizmu uwagi.','',
        '## 3. Strojenie pojedynczych jednostek','',
        'Zapisano AUROC aktywacji każdego neuronu względem etykiety każdego zadania przy trzech poziomach szumu (`task_label_tuning.csv`). Dodatkowo wykonano kontrolowane sondy Posnera: cel/dystraktor tylko w jednej lokalizacji, z ramką i bez niej, oraz sama ramka vs tło. Po 512 prób na klasę, SD=0,08, niezależny szum.','',
        '`posner_tuning_probes.csv` zawiera AUROC dla celu, wskazówki oraz zmianę bezwzględnej czułości na cel przy dodaniu ramki. To opisowe sondy. Nie zastosowano klasyfikacji typów neuronów opartej na istotności/FDR ani pełnego zestawu bodźców z PNAS. Te wyniki nie uprawniają jeszcze do przypisania jednostek do mechanizmów location-opponent / summation.','',
        '## Kontrole i zakres wniosków','',
        '- Zgodność obliczeń na zapisanych aktywacjach z rzeczywistym forward hookiem sprawdzono liczbowo; wynik w `verification.json`.',
        '- Testy sprawdzają interwencję end-to-end, brak informacji o obrazie po wyłączeniu całej warstwy oraz korektę wielokrotnych porównań.',
        '- Grupy wybrano na walidacji i zamrożono przed oceną potwierdzającą. Nie wybierano najlepszego k ani najlepszego poziomu szumu na teście.',
        '- Analiza dotyczy jednej sieci, jednej warstwy oraz obecnej adaptacji bodźców 64×64. ID jednostek nie przenoszą się między treningami. Do wniosku o powtarzalnym mechanizmie potrzebne są niezależne sieci i dalsze analizy warstw konwolucyjnych.',
        '- Lezje wskazują przyczynowy wpływ aktywacji w tym modelu i przy tych interwencjach. Nie dowodzą analogicznej organizacji biologicznej ani tego, że istnieje tylko jeden taki zestaw jednostek.','',
        '## Wykresy','', '![Task lesions](task_lesions.png)','', '![Dose response](lesion_dose.png)','',
        '![Readout controls](readout_controls.png)','', '![Attention lesions](attention_lesions.png)','', '![AUROC probes](tuning_probes.png)','',
        '## Odtworzenie','', 'Z katalogu głównego projektu:', '', '```bash',
        'python3 -m unittest discover -s small_cnn -p "test_lesion.py" -v',
        'python3 small_cnn/lesion.py', 'python3 small_cnn/lesion_attention.py',
        'python3 small_cnn/lesion_checks.py', 'python3 small_cnn/lesion_readout_controls.py', 'MPLCONFIGDIR=tmp/matplotlib python3 small_cnn/summarize_lesion.py','```','',
        'Surowe wyniki: `results.json`, `single_unit_lesions.csv`, `selected_groups.json`, `attention_results.json`, `cross_replacement_checks.json`, `readout_controls.json`. Aktywacje i metadane pozwalają odtworzyć analizy bez ponownego przejścia przez warstwy konwolucyjne.']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('Wrote',R/'REPORT.md')
