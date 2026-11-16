"""Re-publish an existing completed s2 QB campaign; never run training."""
import argparse
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',choices=('original','c30','repeat'),required=True)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    from .release import verify
    verify()
    if a.campaign=='c30':
        from panda_input3_c30.install import install
        install()
    elif a.campaign=='repeat':
        from panda_input3_repeat.install import install
        install()
    from panda_input3 import plan
    if (plan.ROOT/'gspread/server.txt').read_text().strip()!='s2':
        raise ValueError('Read and publish the original files on physical s2')
    root=plan.campaign_root('s2')
    if not (root/'campaign_state.json').exists():raise ValueError('Source campaign is missing')
    # Recover only after its existing runner has released the run lock.
    import fcntl
    with (root/'runner.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from .reporting import install,publish
        install()
        print(json.dumps(publish(root,'s2',apply=a.apply),ensure_ascii=False))


if __name__=='__main__':main()
