"""Read-only client for the JCU Synology Drive share (Vizina/Brdy AudioMoth 2025, ~21 TB).

The share link was provided by the data owner for dialect training.  Only list/download calls are used.
  python jcu_nas.py manifest <out.csv>                -> full recursive listing (folder, name, size, mtime, file_id)
  python jcu_nas.py get <file_id> <out_path>          -> download one file (streamed, resumable via temp file)

The share address and its access token are NOT stored in this (public) repository. Get them from the
project owner and set, before use:
  JCU_NAS_HOST         https://<host>:<port> of the share
  JCU_NAS_LINK_ID      the id in the share link
  JCU_NAS_LINK_TOKEN   the token in the share link
  JCU_NAS_ROOT_ID      Synology Drive file id of the share's root folder
"""

import os, sys, json, time, csv, http.cookiejar, urllib.request, urllib.parse, ssl

HOST = os.environ.get('JCU_NAS_HOST', '')
LINK_ID = os.environ.get('JCU_NAS_LINK_ID', '')
LINK_TOKEN = os.environ.get('JCU_NAS_LINK_TOKEN', '')
ROOT_ID = os.environ.get('JCU_NAS_ROOT_ID', '')


class Share:
    def __init__(self):
        if not (HOST and LINK_ID and LINK_TOKEN and ROOT_ID):
            raise SystemExit(
                'set JCU_NAS_HOST, JCU_NAS_LINK_ID, JCU_NAS_LINK_TOKEN and JCU_NAS_ROOT_ID'
            )
        self.cj = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cj),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        )
        self.op.addheaders = [('User-Agent', 'Mozilla/5.0 (research data download)')]
        self.op.open(f'{HOST}/d/s/{LINK_ID}/{LINK_TOKEN}', timeout=60).read()
        self.tok = next(c.value for c in self.cj if c.name.startswith('drive-sharing'))

    def api(self, params, stream=False, timeout=120):
        data = urllib.parse.urlencode({**params, 'sharing_token': json.dumps(self.tok)}).encode()
        req = urllib.request.Request(
            f'{HOST}/webapi/entry.cgi', data=data, headers={'X-SYNO-SHARING': self.tok}
        )
        r = self.op.open(req, timeout=timeout)
        return r if stream else json.loads(r.read())

    def list(self, fid):
        out, off = [], 0
        while True:
            for attempt in range(5):
                try:
                    d = self.api(
                        {
                            'api': 'SYNO.SynologyDrive.Files',
                            'method': 'list',
                            'version': 2,
                            'path': json.dumps(f'id:{fid}'),
                            'offset': off,
                            'limit': 1000,
                            'sort_by': '"name"',
                            'sort_direction': '"asc"',
                        }
                    )
                    break
                except Exception as e:
                    time.sleep(5 * (attempt + 1))
            items = d['data']['items']
            out += items
            if len(items) < 1000:
                return out
            off += 1000

    def download(self, fid, out_path):
        tmp = out_path + '.part'
        r = self.api(
            {
                'api': 'SYNO.SynologyDrive.Files',
                'method': 'download',
                'version': 2,
                'files': json.dumps([f'id:{fid}']),
                'force_download': 'true',
            },
            stream=True,
            timeout=600,
        )
        with open(tmp, 'wb') as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(tmp, out_path)
        return os.path.getsize(out_path)


def manifest(out):
    s = Share()
    rows = []
    stack = [('', ROOT_ID)]
    while stack:
        path, fid = stack.pop()
        for it in s.list(fid):
            p = f'{path}/{it["name"]}'
            if it['content_type'] == 'dir':
                stack.append((p, it['file_id']))
            else:
                rows.append(
                    dict(
                        path=p,
                        name=it['name'],
                        size=it.get('size', 0),
                        mtime=it.get('modified_time', 0),
                        file_id=it['file_id'],
                    )
                )
        print(path or '/', len(rows), flush=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'name', 'size', 'mtime', 'file_id'])
        w.writeheader()
        w.writerows(rows)
    print('files', len(rows), 'TB', sum(r['size'] for r in rows) / 1e12)


if __name__ == '__main__':
    if sys.argv[1] == 'manifest':
        manifest(sys.argv[2])
    elif sys.argv[1] == 'get':
        print(Share().download(sys.argv[2], sys.argv[3]))
