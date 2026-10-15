"""Read-only lookup of PAN archive locations; no SSH or payload writes."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Find the recorded storage location of a PAN run or runtime.')
    p.add_argument('query',nargs='?',default='')
    p.add_argument('--server',choices=['s1','s2','s3','s4','s5'])
    p.add_argument('--json',action='store_true')
    a=p.parse_args()
    index=json.loads(Path(__file__).with_name('ARCHIVE_INDEX.json').read_text(encoding='utf-8'))
    found=[]
    for entry in index['entries']:
        if a.server and entry['storage_server']!=a.server:continue
        hay=' '.join(str(entry.get(k,'')) for k in ['run_id','source_path','storage_path','kind']).lower()
        if a.query.lower() not in hay:continue
        item=dict(entry)
        if item['storage_server']==index['recording_server']:
            item['storage_path_exists_now']=Path(item['storage_path']).exists()
        else:item['storage_path_exists_now']='not_checked_remote_server'
        found.append(item)
    if a.json:print(json.dumps(found,ensure_ascii=False,indent=2))
    else:
        print('Recorded locations: '+str(len(found))+'; index date: '+index['updated_date'])
        for e in found:
            print('\n['+e['storage_server']+'] '+e['kind']+' '+e['run_id'])
            print('  source : '+e['source_path'])
            print('  stored : '+e['storage_path'])
            print('  volume : '+str(e.get('volume_uuid')))
            print('  state  : '+e['verification'])
            print('  exists : '+str(e['storage_path_exists_now']))
            if e.get('file_manifest'):print('  manifest: '+e['file_manifest'])
        if not found:print('No indexed match. This does not mean the run is absent or safe to delete.')
    return 0 if found else 1

if __name__=='__main__':raise SystemExit(main())
