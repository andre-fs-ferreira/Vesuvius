import sys
import os
sys.path.append(os.path.abspath("/home/shadowtwin/Desktop/AI_work/Vesuvius_Challenge/topological-metrics-kaggle/src"))
import topometrics.leaderboard
import glob
import importlib
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
tqdm.pandas()

class VesuviusMetric():
    """
    Vesuvius competition metric.

    Expects standard topometrics to be installed. Download from https://www.kaggle.com/datasets/sohier/vesuvius-metric-resources/data
    """
    def __init__(self, solution_path=None, submission_path=None, output_file=None):
        self.solution_path = solution_path
        self.submission_path = submission_path
        self.output_file = output_file

    def load_volume(self, path):
        im = Image.open(path)
        slices = []
        for i, page in enumerate(ImageSequence.Iterator(im)):
            slice_array = np.array(page)
            slices.append(slice_array)
        volume = np.stack(slices, axis=0)
        return volume


    def score_single_tif(
        self,
        gt_path,
        pred_path,
        surface_tolerance,
        voi_connectivity=26,
        voi_transform='one_over_one_plus',
        voi_alpha=0.3,
        topo_weight=0.3,
        surface_dice_weight=0.35,
        voi_weight=0.35,
        ):
        gt: np.ndarray = self.load_volume(gt_path)
        pr: np.ndarray = self.load_volume(pred_path)

        # The import is here to ensure dependencies are loaded first.
        score_report = topometrics.leaderboard.compute_leaderboard_score(
            predictions=pr,
            labels=gt,
            dims=(0, 1, 2),
            spacing=(1.0, 1.0, 1.0),  # (z, y, x)
            surface_tolerance=surface_tolerance,  # in spacing units
            voi_connectivity=voi_connectivity,
            voi_transform=voi_transform,
            voi_alpha=voi_alpha,
            combine_weights=(topo_weight, surface_dice_weight, voi_weight),  # (Topo, SurfaceDice, VOI)
            fg_threshold=None,  # None => legacy "!= 0"; else uses "x > threshold"
            ignore_label=2,  # voxels with this GT label are ignored
            ignore_mask=None,  # or pass an explicit boolean mask
        )
        
        return {
            'image_score': np.clip(score_report.score, 0.0, 1.0),
            'topo_score': score_report.topo.toposcore,
            'surface_dice': score_report.surface_dice,
            'voi_score': score_report.voi.voi_score,
            'voi_split': score_report.voi.voi_split,
            'voi_merge': score_report.voi.voi_merge
        }

    def score(
        self,
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        row_id_column_name: str,
        surface_tolerance: float = 2.0,
        voi_connectivity: int = 26,
        voi_transform: str = 'one_over_one_plus',
        voi_alpha: float = 0.3,
        topo_weight: float = 0.3,
        surface_dice_weight: float = 0.35,
        voi_weight: float = 0.35,
        ) -> float:
        """Returns the mean per-volume scores and attaches sub-metrics to the solution df."""

        solution['pred_paths'] = submission['tif_paths']
        
        # Apply the scoring function and expand the returned dict into new columns
        metrics_df = solution.progress_apply(
            lambda row: self.score_single_tif(
                row['tif_paths'],
                row['pred_paths'],
                surface_tolerance,
                voi_connectivity=voi_connectivity,
                voi_transform=voi_transform,
                voi_alpha=voi_alpha,
                topo_weight=topo_weight,
                surface_dice_weight=surface_dice_weight,
                voi_weight=voi_weight,
            ),
            axis=1,
        ).apply(pd.Series)

        # Merge the new metric columns back into the original solution dataframe
        solution = pd.concat([solution, metrics_df], axis=1)

        # Return the mean of the primary leaderboard score
        return float(np.mean(solution['image_score'])), solution

    def _run(self):
        # 1. Load the solution and submission dataframes
        solution = pd.read_csv(self.solution_path)
        submission = pd.read_csv(self.submission_path)

        # 2. Call the score function
        # Note: 'row_id_column_name' should match the ID column in your CSV (e.g., 'id' or 'segment_id')
        final_score, solution = self.score(
            solution=solution,
            submission=submission,
            row_id_column_name='id'
        )

        # Calculate the Mean for all numeric columns
        # We select only numeric types so we don't try to average the file paths
        metrics_only = solution.select_dtypes(include=[np.number])
        mean_series = metrics_only.mean()
        
        # Create a "Mean" row
        # We use .copy() to avoid SettingWithCopy warnings
        mean_row = pd.DataFrame(mean_series).get_values().T if hasattr(mean_series, 'get_values') else pd.DataFrame([mean_series])
        
        # Label the ID column as "MEAN" for clarity
        mean_row['id'] = 'MEAN'
        
        # 4. Append the mean row to the bottom
        solution = pd.concat([solution, mean_row], ignore_index=True)

        # 5. Save to disk
        solution.to_csv(self.output_file, index=False)
        
        print("\n" + "="*30)
        print("📊 FINAL AGGREGATED METRICS")
        print("="*30)
        # Display the last row (the mean row) in a nice format
        print(solution.iloc[-1:][metrics_only.columns].to_string(index=False))
        print("="*30)
        print(f"✅ Saved detailed metrics with Mean row to: {self.output_file}")