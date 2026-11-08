"""Create figures and a Polish result sheet from completed measurements."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent/'results'
m=json.loads((R/'metrics.json').read_text())
h=json.loads((R/'history.json').read_text())

def row(noise,task,condition='all'):
    return next(r for r in m['evaluations'][str(float(noise))] if r['task']==task and r['condition']==condition)

fig,axs=plt.subplots(1,3,figsize=(13,3.5))
axs[0].plot([a['epoch'] for a in h],[a['train_accuracy'] for a in h],label='train (mixed noise)')
axs[0].plot([a['epoch'] for a in h],[a['val_accuracy'] for a in h],label='validation (SD .04)')
axs[0].set(xlabel='Epoch',ylabel='6-class accuracy',ylim=(.3,1)); axs[0].legend(fontsize=8)
for task in ['posner','search','context']:
    ns=[float(n) for n in m['evaluations']]
    vals=[row(n,task)['accuracy'] for n in ns]
    axs[1].plot(ns,vals,'o-',label=task)
axs[1].set(xlabel='Noise SD after normalization',ylabel='Test accuracy',ylim=(.45,1.02)); axs[1].legend(fontsize=8)
for condition in ['old','new']:
    ns=[float(n) for n in m['evaluations']]
    axs[2].plot(ns,[row(n,'context',condition)['accuracy'] for n in ns],'o-',label=condition)
axs[2].set(xlabel='Noise SD',ylabel='Context orientation accuracy',ylim=(.45,1.02)); axs[2].legend()
for ax in axs: ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(R/'performance.png',dpi=170); plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(12,3.5))
for noise in [.04,.08,.14]:
    axs[0].plot(['valid','invalid'],[row(noise,'posner',c)['hit_rate'] for c in ['valid','invalid']],'o-',label=f'SD={noise}')
    axs[1].plot([1,2,4,8],[row(noise,'search',f'set_{s}')['accuracy'] for s in [1,2,4,8]],'o-',label=f'SD={noise}')
    axs[2].plot(['old','new'],[row(noise,'context',c)['accuracy'] for c in ['old','new']],'o-',label=f'SD={noise}')
axs[0].set(ylabel='Posner hit rate'); axs[1].set(xlabel='Number of cue boxes',ylabel='Search accuracy'); axs[2].set(ylabel='Context accuracy')
for ax in axs: ax.set_ylim(.3,1.02); ax.legend(fontsize=8); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(R/'attention_effects.png',dpi=170); plt.close(fig)

lines=['# Wyniki jednej małej CNN dla trzech zadań','',
       f"Model: **{m['parameters']:,} parametrów**, najlepsza epoka {m['best_epoch']}; pełny trening {m['training_seconds']/60:.1f} min na CPU.",
       '', 'Checkpoint wybrano wyłącznie na walidacji przy SD=0,04. Test: 4 000 prób na zadanie (12 000 ogółem).', '',
       '## Trafność na odłożonym zbiorze testowym', '',
       'Głównym warunkiem jest SD=0,04, ustalone przed treningiem. Decyzja spośród sześciu klas; przypadkowa decyzja przy znanym zadaniu to 50%.', '',
       '| Zadanie | Accuracy | 95% CI Wilsona | Accuracy przy znanym zadaniu |', '|---|---:|---:|---:|']
for task in ['posner','search','context']:
    a=row(.04,task); lo,hi=a['ci95']
    lines.append(f"| {task} | {a['accuracy']:.2%} | {lo:.2%}–{hi:.2%} | {a['task_given_accuracy']:.2%} |")
lines+=['', 'Dla Posnera optymistyczny obserwator z bezszumowym dostępem do ukrytych kątów i strony wskazówki osiąga około 84,64% (milion prób, `angle_oracle.py`). Nakładanie rozkładów kątów wyklucza oczekiwanie 100% trafności. To inny benchmark niż obrazowy BIO z publikacji.', '', '## Odporność na szum','','| SD | Posner | Search | Context |','|---|---:|---:|---:|']
for noise in [0,.04,.08,.14,.20]:
    lines.append('| '+str(noise)+' | '+' | '.join(f"{row(noise,t)['accuracy']:.2%}" for t in ['posner','search','context'])+' |')
lines+=['','## Efekty uwagowe','','Nie utożsamiamy hit rate z accuracy: valid i invalid dotyczą tylko prób z celem. False alarm rate liczymy osobno na próbach bez celu. Wyniki Posnera poniżej agregują cztery kontrasty ramki, także zerowy.', '',
        '| SD | Posner HR valid | Posner HR invalid | FAR | Search SS1 | SS2 | SS4 | SS8 | Context old | new |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for noise in [.04,.08,.14,.20]:
    vals=[row(noise,'posner','valid')['hit_rate'],row(noise,'posner','invalid')['hit_rate'],row(noise,'posner','absent')['false_alarm_rate']]
    vals += [row(noise,'search',f'set_{s}')['accuracy'] for s in [1,2,4,8]]
    vals += [row(noise,'context',c)['accuracy'] for c in ['old','new']]
    lines.append('| '+str(noise)+' | '+' | '.join(f'{v:.2%}' for v in vals)+' |')
lines+=['', 'W tej jednej sieci występują wszystkie trzy oczekiwane tendencje behawioralne:', '',
        f"- Posner, SD=0,04: hit rate valid {row(.04,'posner','valid')['hit_rate']:.2%}, invalid {row(.04,'posner','invalid')['hit_rate']:.2%}, FAR {row(.04,'posner','absent')['false_alarm_rate']:.2%}.",
        f"- Search, SD=0,04: accuracy dla set size 1: {row(.04,'search','set_1')['accuracy']:.2%}; dla set size 8: {row(.04,'search','set_8')['accuracy']:.2%}.",
        f"- Context, SD=0,14: old {row(.14,'context','old')['accuracy']:.2%}, new {row(.14,'context','new')['accuracy']:.2%}. Przy SD=0,04: old {row(.04,'context','old')['accuracy']:.2%}, new {row(.04,'context','new')['accuracy']:.2%}. Sufit trafności może ukrywać efekt kontekstu.", '',
        'To opis obserwowanych różnic. Nie przeprowadzono testu istotności uwzględniającego klastrowanie starych kontekstów ani powtórzeń treningu. Warunek SD=0,14 pokazano jako jeden z wcześniej ustalonych poziomów szumu; nie jest nowym kryterium wyboru modelu.', '',
        '## Zakres wniosków','',
        'To pomiary na wygenerowanych danych w jawnie opisanej adaptacji procedury, a nie pełna replikacja publikacji ani wynik na gotowym oryginalnym zbiorze obrazów. Rozdzielczość 64×64, antyaliasing, uproszczenia geometrii i mieszanka szumu zmieniają trudność. Wynik przy SD=0,04 nie oznacza wyniku przy szumie publikacji. Jedna inicjalizacja nie wystarcza do oceny stabilności efektów między sieciami.', '',
        'Przedziały Wilsona są opisowe dla jednej sieci; powtarzane konteksty nie są niezależnymi konfiguracjami. Nie przeprowadzono walidacji neuronalnej ani dopasowania do danych ludzi, więc wysoka trafność nie potwierdza mechanizmów pojedynczych jednostek z PNAS. Należy patrzeć osobno na odczyt orientacji/detekcję i różnice między warunkami.', '',
        'Stare tła są świadomie wspólne między treningiem a testem, ale szum jest nowy, a etykiety orientacji zbalansowane. Nowe tła są rozłączne między splitami. `split_audit.json` dokumentuje to rozróżnienie. Surowe predykcje zawierają metadane każdej próby.', '',
        'Wyniki testowe nie były używane do wyboru checkpointu. Szczegóły celu prac, danych i odstępstw: `../ANALIZA.md`.', '',
        '![Training and noise](performance.png)', '', '![Condition effects](attention_effects.png)']
(R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:18]))
