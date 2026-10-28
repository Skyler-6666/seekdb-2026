"""Generate optional Parquet regression inputs; Arrow only writes test fixtures."""
import argparse
import datetime
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def generate(directory):
    directory.mkdir(parents=True, exist_ok=False)
    table = pa.table({
        'id': pa.array([-1, 0, None, 2147483647, 9223372036854775807], type=pa.int64()),
        'small': pa.array([1, None, -3, 4, 5], type=pa.int32()),
        'text': ['a,b', '你好', None, 'a,b', 'line\nend'],
        'score': pa.array([1.25, -2.0, 3.5, None, 5.5], type=pa.float64()),
        'active': [True, False, None, True, False],
        'day': pa.array([datetime.date(2024,2,29), None, datetime.date(1970,1,1), datetime.date(2026,8,1), datetime.date(2000,1,1)], type=pa.date32()),
        'created': pa.array([datetime.datetime(2026,1,1,1,2,3,123456), None, datetime.datetime(1970,1,1), datetime.datetime(1969,12,31,23,59,59,999999), datetime.datetime(2000,1,1)], type=pa.timestamp('us')),
    })
    cases = {}
    def save(name, table, **options):
        pq.write_table(table, directory / name, **options)
        records = table.to_pydict()
        cases[name] = [[records[column][i] for column in table.column_names] for i in range(table.num_rows)]
    for compression in ('NONE','SNAPPY'):
        for version in ('1.0','2.0'):
            for dictionary in (False,True):
                save(f'fixture-{compression}-{version}-{dictionary}.parquet', table,
                     compression=compression, data_page_version=version, use_dictionary=dictionary,
                     row_group_size=2, write_page_checksum=True)
    integers = [-9223372036854775808,-1,0,9223372036854775807,None] + list(range(1000))
    names = ['prefix-' + str(i//2) if i%5 else None for i in range(len(integers))]
    delta = pa.table({'id':pa.array(integers,type=pa.int64()),'name':names,'score':[i/8 for i in range(len(integers))]})
    for version in ('1.0','2.0'):
        for encoding in ('DELTA_LENGTH_BYTE_ARRAY','DELTA_BYTE_ARRAY'):
            save(f'encoding-{encoding}-{version}.parquet', delta, compression='SNAPPY',
                 use_dictionary=False, column_encoding={'id':'DELTA_BINARY_PACKED','name':encoding,'score':'BYTE_STREAM_SPLIT'},
                 row_group_size=211, data_page_size=128, data_page_version=version)
    def scalar(value):
        if isinstance(value, datetime.datetime): return value.isoformat(sep=' ')
        if isinstance(value, datetime.date): return value.isoformat()
        raise TypeError(type(value).__name__)
    (directory/'expected.json').write_text(json.dumps(cases, ensure_ascii=False, default=scalar), encoding='utf-8')
    dictionary = pa.DictionaryArray.from_arrays(pa.array([0]*1000, type=pa.int32()), pa.array(['x'*(8*1024*1024)]))
    pq.write_table(pa.table({'text':dictionary}), directory/'dictionary-expansion.parquet',
                   compression='SNAPPY', use_dictionary=True, write_batch_size=1000, data_page_size=1024*1024)
    print(f'Generated {len(cases)} Parquet fixtures and their expected rows in {directory}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='new directory beneath secure_file_priv')
    generate(parser.parse_args().data_dir)
