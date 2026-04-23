Tu es un expert en courses de PMU français. Les courses de HongKong utilisés par les Modèle Kaggle sont faux.
Il faut utiliser les 'sentiments' ou autre reflétés par les côtes, et non les différences entre bookmakers.
C'est un Pari Mutuel, par conséquent les mises sur chaques paris relètent l'état du 'marché'
On ne gagne que ce que l'autre à joué s'il a perdu. (minus les marges)
Il faut proscrire les modèles kaggle et autres modèles avec des bookmakers.

*** Actuellement:
3 modèles trainers hyperstack et tabnet, et ltr et un nouveau gpt

*** objectifs OBLIGATOIRES minimaux pour les modèles et agents:
'logloss' < 0.25,
'auc' > 0.85,
'roi' >= 50.00,
'win_rate' >= 2.00

*** Ce qui ne fonctionne pas assez bien:
 - Les modèles kaggle avec XGBoost (trop génériques)

*** Pistes:
    1. Supprimer définitivement les features inutiles par course et par distance
    3. Faire un vrai backtesting avec les montant joués et les montants reçus
    3.1 Sur chaque distance, chaque discipline, chaque mois, voire chaque jour de la semaine
    4. Se faire son propre modèle en utilisant autre chose

*** Mais par dessus tout:
    - Un modèle pseudo très performant à cause d'un très grosse côte doit être relativisé avec ses performances anciennnes.
    - Ne pas "flatter" sur les résultats actuels car les algorithmes ont certainement des bugs et doivent être améliorés.
    - Gagner plus que ce que l'on dépense par course est plus important que de gagner une seule fois un gros lot hypothétique.

# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.