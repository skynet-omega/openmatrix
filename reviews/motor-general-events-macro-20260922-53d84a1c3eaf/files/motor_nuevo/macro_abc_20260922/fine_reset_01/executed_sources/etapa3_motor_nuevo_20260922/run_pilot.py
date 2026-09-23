"""Current stage3 entrypoint; engine choice is mandatory and recorded.

The interrupted historical runner and every executed source copy are preserved.
This entrypoint does not restart the cancelled queue or infer stage3 admission.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'motor_nuevo/pipeline_review_20260922'))
from run_pipeline import main
if __name__=='__main__':raise SystemExit(main())
