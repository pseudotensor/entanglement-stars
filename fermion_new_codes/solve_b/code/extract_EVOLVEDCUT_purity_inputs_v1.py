"""Select frozen matched-partition metadata; no eigensolver or entropy computation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cpu_runtime import require_remote_cpu

MATCHES=[(12,0,58),(15,1,51),(20,1,38),(28,1,37),(35,1,37),(40,1,36)]


def read_clock_table(text):
    section=text.split('### Clock table (weighted, 16 kappa, kappa <= 10)',1)[1].split('\n### ',1)[0]
    rows={}
    for line in section.splitlines():
        cells=[c.strip() for c in line.split('|')[1:-1]]
        if not cells or not cells[0].isdigit():continue
        clock=int(cells[0])
        if clock in rows or len(cells)!=12:raise ValueError('duplicate or malformed printed clock row')
        rows[clock]=cells
    if set(rows)!={12,15,16,20,24,25,28,30,35,40}:raise ValueError('missing printed clock row')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--first-table',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();runtime=require_remote_cpu(args.output)
    raw=args.input.read_bytes();saved=json.loads(raw);table_bytes=args.first_table.read_bytes();printed=read_clock_table(table_bytes.decode())
    rows=[];first_rows=[]
    for clock,shift,sites in MATCHES:
        records=sorted([r for r in saved['rows'] if r['clock_rs']==clock],key=lambda r:r['kappa'])
        if [r['kappa'] for r in records]!=list(range(1,11)):raise ValueError('missing or duplicate channel')
        if sum(r['weight'] for r in records)!=880:raise ValueError('incorrect multiplicity')
        if printed[clock][5]!=f'shift +{shift} ({sites} sites)':raise ValueError('printed matched partition differs')
        first_rows.append({'clock_rs':clock,'site_shift':shift,'interior_sites':sites,
                           'printed_excess_decimal':printed[clock][6],'source_cells':printed[clock]})
        for record in records:
            cut=[c for c in record['cuts'] if c['site_shift']==shift]
            if len(cut)!=1 or cut[0]['interior_sites']!=sites:raise ValueError('saved matched partition differs')
            cut=cut[0];polar=cut['orthonormal_control'];checks=record['spectral_checks']
            if record['source_dtype']!=printed[clock][1] or record['weight']!=16*record['kappa']:raise ValueError('input channel metadata mismatch')
            if not polar['interior']['valid'] or not cut['held']['valid']:raise ValueError('unavailable saved entropy')
            rank=record['dust_rank']
            if rank and (record['occupation_min']!=1 or record['occupation_max']!=1):raise ValueError('nonbinary dust occupations')
            block_checks={key:checks[key] for key in [f'orthonormal_control_interior_shift{shift}',f'held_shift{shift}',f'flat_shift{shift}']}
            if any(c['dimension']!=2*sites or c['eigenpair_frobenius_residual'] is None or c['eigenvector_orthogonality_frobenius'] is None or c['eigvalsh_eigh_max_difference'] is None for c in block_checks.values()):raise ValueError('missing small-block diagnostics')
            folder=args.input.parent/f'T{clock}_k{record["kappa"]}'
            held_path=folder/'held_operator_eigenvalues.npy'
            held_values=np.load(held_path,allow_pickle=False)
            if held_values.shape!=(4096,) or held_values.dtype!=np.float64 or not np.all(np.isfinite(held_values)):raise ValueError('invalid saved full held spectrum')
            held_negative=int(np.count_nonzero(held_values<0));held_gap=float(np.min(np.abs(held_values)))
            if held_negative!=record['held_operator']['negative_count'] or held_gap!=record['held_operator']['nearest_zero']:raise ValueError('held spectrum/report binding differs')
            held_metadata={'source':str(held_path),'sha256':hashlib.sha256(held_path.read_bytes()).hexdigest(),
                           'dimension':len(held_values),'dtype':str(held_values.dtype),'negative_count':held_negative,
                           'minimum_absolute_eigenvalue':held_gap,'maximum_absolute_eigenvalue':float(np.max(np.abs(held_values)))}
            rows.append({'clock_rs':clock,'kappa':record['kappa'],'weight':record['weight'],
                         'site_shift':shift,'interior_sites':sites,'dimension':2*sites,'source_dtype':record['source_dtype'],
                         'original_state_source':record['source'],'record_source':str(folder/'result.json'),
                         'polar_spectra_source':str(folder/f'orthonormal_control_cut{sites}.npz'),
                         'held_spectrum_source':str(folder/f'held_cut{sites}.npy'),
                         'flat_spectrum_source':str(folder/f'flat_cut{sites}.npy'),
                         'polar_interior':polar['interior'],'polar_physical_interior_entropy':polar['physical_interior_entropy'],
                         'polar_exterior':polar['exterior'],'held':cut['held'],'flat':cut['flat'],
                         'orthogonality':record['orthogonality'],'ideal_polar_global_entropy':record['ideal_polar_global_entropy'],
                         'held_operator':record['held_operator'],'held_spectrum_metadata':held_metadata,
                         'polar_column_count':record['polar_global']['dimension'],'source_row_dimension':4096,
                         'flat_operator':record['flat_operator'],
                         'small_block_checks':block_checks,'raw_global_valid':record['raw_global']['valid'],
                         'dust_rank':rank,'binary_occupations_confirmed':True})
    answer={'scope':'Saved scalar metadata only. Only saved-spectrum extrema/counts added; no eigensolver, entropy, covariance, residual or error bound recomputed.',
            'runtime':runtime,'source':str(args.input),'source_sha256':hashlib.sha256(raw).hexdigest(),
            'first_table_source':str(args.first_table),'first_table_sha256':hashlib.sha256(table_bytes).hexdigest(),
            'extractor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rows':rows,'first_route_printed_rows':first_rows,
            'controls':{'matched_clock_count':len(MATCHES),'matched_channel_rows':len(rows),'all_small_block_diagnostics_present':True,
                        'all_binary_occupations':True,'exact_partition_dtype_and_multiplicity_bindings':True,'all_full_held_spectrum_dimensions_gap_and_negative_counts_bound':True}}
    (args.output/'matched_input_metadata.json').write_text(json.dumps(answer,indent=2,allow_nan=False)+'\n')
    print(json.dumps(answer['controls']),flush=True)


if __name__=='__main__':main()
