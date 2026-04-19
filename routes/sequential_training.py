from flask import Blueprint, request, Response
import json
import datetime
import os
from analysis.deep_learning_model import DLMacroModel

sequential_bp = Blueprint('sequential', __name__)

@sequential_bp.route('/api/sequential/train', methods=['GET'])
def api_sequential_train_stream():
    """
    SSE endpoint for sequential year-by-year training.
    Query params: start_year (int)
    Trains LSTM models from start_year to current year.
    """
    # Extract request parameters OUTSIDE generator to avoid context issues
    start_year = int(request.args.get('start_year', 2019))
    
    def generate():
        try:
            current_year = datetime.datetime.now().year
            
            # Calculate year range
            years = list(range(start_year, current_year + 1))
            total_years = len(years)
            
            yield f"data: {json.dumps({'type': 'start', 'total_years': total_years, 'years': years})}\n\n"
            
            for idx, year in enumerate(years):
                year_num = idx + 1
                train_cutoff_date = f"{year}-12-31"
                
                # Notify year start
                yield f"data: {json.dumps({'type': 'year_start', 'year': year, 'year_num': year_num, 'total_years': total_years})}\n\n"
                
                # Create output folder
                output_folder = os.path.join('SequentialDLModels', f'{year}-12-31', 'LSTM')
                os.makedirs(output_folder, exist_ok=True)
                
                # Initialize model
                dl_model = DLMacroModel()
                
                # Step 1: Optimization
                yield f"data: {json.dumps({'type': 'step_progress', 'step': 1, 'progress': 10, 'message': f'Optimizing LSTM for {year}...'})}\n\n"
                
                best_params = dl_model.optimize_models(
                    n_trials=10,
                    train_cutoff_date=train_cutoff_date
                )
                
                yield f"data: {json.dumps({'type': 'step_progress', 'step': 1, 'progress': 33, 'message': 'Optimization complete'})}\n\n"
                
                # Step 2 & 3: Discovery + Production Training
                yield f"data: {json.dumps({'type': 'step_progress', 'step': 2, 'progress': 40, 'message': f'Training LSTM for {year}...'})}\n\n"
                
                dl_model.train_all_models(
                    model_type='lstm',
                    train_cutoff_date=train_cutoff_date,
                    output_folder=output_folder,
                    n_folds=1,  # Single fold for sequential
                    max_epochs=50,
                    ensemble_mode=True
                )
                
                yield f"data: {json.dumps({'type': 'step_progress', 'step': 3, 'progress': 100, 'message': f'{year} complete'})}\n\n"
                
                # Year complete
                yield f"data: {json.dumps({'type': 'year_complete', 'year': year, 'year_num': year_num, 'total_years': total_years, 'path': output_folder})}\n\n"
            
            # All done
            yield f"data: {json.dumps({'type': 'done', 'message': 'Sequential training complete!', 'total_years': total_years})}\n\n"
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')
