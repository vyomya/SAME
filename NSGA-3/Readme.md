# NSGA-III Semi-Manual Optimizer

pip install pymoo pandas numpy

────────────────────────────────────────────────────
YOUR CSV COLUMN NAMES (exact match, either format works)
────────────────────────────────────────────────────
Parameters:   xN (Model)  |  xT(Input length)  |  xV(Resolution/Stride)
              xRLora Rank  |  xQ(Quantization)  |  xP(Pruning)

Objectives:   WER  |  FLOPS  |  Space Memory

Rows without all 3 objectives filled = treated as pending (ignored for now).

────────────────────────────────────────────────────
COMMANDS
────────────────────────────────────────────────────

1. SUGGEST — get next N configs to train
   python nsga3_optimizer.py suggest --csv your_results.csv --n 10
   → outputs: suggested_configs.csv  (param cols + empty WER/FLOPS/Memory cols)

2. REGISTER — after you've trained & filled in objectives
   python nsga3_optimizer.py register completed_batch.csv
   → updates: eval_cache.json

3. PARETO — compute Pareto front from everything registered so far
   python nsga3_optimizer.py pareto
   → outputs: pareto_results.csv + prints knee point

────────────────────────────────────────────────────
ITERATIVE WORKFLOW
────────────────────────────────────────────────────
Round 1:  suggest --csv initial.csv --n 10  → train 10 configs
Round 2:  register batch1.csv              → cache grows
          suggest --n 10                   → NSGA-III now guided by data
Round 3+: repeat register → suggest → train
Final:    pareto                           → full Pareto front

────────────────────────────────────────────────────
PARAMETER VALUES SUPPORTED
────────────────────────────────────────────────────
xN_model:        Large-v3 | Medium | Small | Tiny | Large-v3 Distil | Medium Distil | Small Distil
xT_input_length: 1500 Frames | 750 Frames
xV_resolution:   1 | 2
xR_lora_rank:    No Lora | r=8 | r=16 | r=32 | r=64
xQ_quantization: QAT-FP32 | QAT-FP16 | QAT-INT8 | QAT-INT4
xP_pruning:      dense | 2:4 | 1:4

python nsga3_optimizer.py register batch1.csv
python nsga3_optimizer.py suggest --csv your_results.csv --n 10
python nsga3_optimizer.py pareto