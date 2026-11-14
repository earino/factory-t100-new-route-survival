from __future__ import annotations

import io
import re
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from source_io import fetch

PAGE = 'https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIM&QO_fu146_anzr=Nv4%20Pn44vr45'
PANEL_YEARS = list(range(2010, 2021))
PANEL_END_YM = 2020 * 12 + 1
TARGET = 'discontinued'
ID_COL = 'episode_id'
TIME_COL = 'entry_month'
CARRY = ['unique_carrier', 'origin_city_market_id', 'dest_city_market_id']
FEATURES = [
    'distance_miles',
    'entry_calendar_month',
    'carrier_departures_prior_12m',
    'carrier_active_pairs_prior_12m',
    'carrier_route_attrition_prior_12m',
    'origin_market_departures_prior_12m',
    'dest_market_departures_prior_12m',
    'competing_carriers_prior_3m',
    'pair_departures_prior_3m',
    'entry_month_departures',
    'entry_month_seats_per_departure',
    'entry_month_load_factor',
    'carrier_new_at_origin',
    'carrier_new_at_dest',
]
COLUMNS = [ID_COL, TIME_COL] + CARRY + FEATURES + [TARGET]
WINDOWS = {
    'train': ('2011-01-01', '2014-07-01'),
    'eval': ('2015-10-01', '2016-10-01'),
    'holdout': ('2018-01-01', '2019-01-01'),
}

FORM_SPEC = {
    'carrier': ('uniquecarrier',),
    'origin_market': ('origincitymarketid', 'origincitymarket', 'originmarketid'),
    'dest_market': ('destcitymarketid', 'destcitymarket', 'destinationcitymarketid'),
    'year': ('year',),
    'month': ('month',),
    'class': ('class', 'serviceclass'),
    'dep': ('departuresperformed', 'depperformed'),
    'seats': ('seats', 'availableseats'),
    'passengers': ('passengers', 'passengerstransported'),
    'distance': ('distance', 'distancebetweenairportsmiles'),
}
CSV_SPEC = FORM_SPEC


def _norm(s):
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())


def _ym(y, m):
    return int(y) * 12 + int(m) - 1


def _split_for_ym(ym):
    if _ym(2011, 1) <= ym <= _ym(2014, 6):
        return 'train'
    if _ym(2015, 10) <= ym <= _ym(2016, 9):
        return 'eval'
    if _ym(2018, 1) <= ym <= _ym(2018, 12):
        return 'holdout'
    return None


def _add(checks, name, observed, expected, passed, kind=None, detail=None):
    rec = {
        'check': name,
        'observed': str(observed)[:3000],
        'expected': str(expected)[:3000],
        'passed': bool(passed),
    }
    if kind is not None:
        rec['kind'] = kind
    checks.append(rec)
    if not passed:
        msg = f'{name} failed. Detail: {detail}. Observed: {observed}. Expected: {expected}.'
        raise AssertionError(msg[:4000])
    return rec


def _is_zip(path):
    try:
        with open(path, 'rb') as fh:
            return fh.read(2) == b'PK'
    except OSError:
        return False


def _new_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    })
    return session


def _read_form(session):
    resp = session.get(PAGE, timeout=240)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'lxml')
    hidden = {}
    boxes = {}
    for inp in soup.find_all('input'):
        name = inp.get('name')
        if not name:
            continue
        typ = (inp.get('type') or 'text').lower()
        if typ == 'hidden':
            hidden[name] = inp.get('value') or ''
        elif typ == 'checkbox':
            boxes[name] = inp.get('value') or 'on'
    selects = {}
    for sel in soup.find_all('select'):
        name = sel.get('name')
        if not name:
            continue
        opts = []
        for opt in sel.find_all('option'):
            val = opt.get('value')
            opts.append(val if val is not None else (opt.get_text() or '').strip())
        selects[name] = opts
    return {'hidden': hidden, 'boxes': boxes, 'selects': selects}


def _refresh(session, state):
    try:
        fresh = _read_form(session)
    except Exception:
        return
    state['hidden'] = fresh['hidden']
    state['boxes'] = fresh['boxes']
    state['selects'] = fresh['selects']


def _select_name(selects, candidates):
    norm = {}
    for name in selects:
        norm[_norm(name)] = name
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    return None


def _resolve_form_fields(boxes, checks):
    norm = {}
    for name in boxes:
        norm[_norm(name)] = name
    chosen = {}
    missing = []
    for key, cands in FORM_SPEC.items():
        hit = None
        for cand in cands:
            if cand in norm:
                hit = norm[cand]
                break
        chosen[key] = hit
        if hit is None:
            missing.append(key)
    observed = sorted(boxes)
    head = observed[:80]
    _add(
        checks,
        'form_field_names',
        f'checkbox_count {len(boxes)} names {head}',
        f'download form exposes field checkboxes for {sorted(FORM_SPEC)}',
        not missing,
        detail=f'missing field keys {missing} observed checkbox names {head}',
    )
    return chosen


def _post_select(session, state, year, period, cache):
    path = cache / f't100d_fim_{year}_{period}.zip'
    if _is_zip(path):
        return path, 'cached archive reused'
    last = 'no attempt'
    for attempt in range(3):
        payload = dict(state['hidden'])
        payload['txtSearch'] = ''
        if state.get('geo_field'):
            payload[state.get('geo_field')] = state.get('geo_value') or 'All'
        payload[state['year_field']] = str(year)
        payload[state['period_field']] = str(period)
        payload['btnDownload'] = 'Download'
        payload['chkDownloadZip'] = 'on'
        for nm in state['fields'].values():
            if nm:
                payload[nm] = 'on'
        try:
            resp = session.post(
                PAGE,
                data=payload,
                timeout=900,
                headers={'Referer': PAGE, 'Origin': 'https://www.transtats.bts.gov'},
            )
        except Exception as exc:
            last = f'session error {type(exc).__name__} {str(exc)[:300]}'
            _refresh(session, state)
            time.sleep(2 + attempt * 2)
            continue
        body = resp.content
        if body[:2] == b'PK':
            path.write_bytes(body)
            return path, f'zip response status {resp.status_code} bytes {len(body)}'
        ctype = resp.headers.get('Content-Type', '')
        last = f'http {resp.status_code} content type {ctype} body start ' + repr(body[:600])
        _refresh(session, state)
        time.sleep(2 + attempt * 2)
    return None, last


def _download_year(session, state, year, cache):
    if year != 2020 and state.get('all_value') is not None:
        path, _note = _post_select(session, state, year, state['all_value'], cache)
        if path is not None:
            return [path]
        _refresh(session, state)
    months = [1, 2] if year == 2020 else list(range(1, 13))
    paths = []
    for month in months:
        path, note = _post_select(session, state, year, month, cache)
        if path is None:
            msg = f'T-100 archive download failed for year {year} month {month}. Last observed response: {note}'
            raise AssertionError(msg[:4000])
        paths.append(path)
    return paths


def _sniff_sep(zf, member):
    with zf.open(member) as fh:
        raw = fh.read(4096).decode('utf-8-sig', errors='ignore')
    first = raw.splitlines()[0] if raw else ''
    cands = [(',', first.count(',')), (chr(9), first.count(chr(9))), ('|', first.count('|'))]
    best = None
    best_count = -1
    for sep, count in cands:
        if count > best_count:
            best = sep
            best_count = count
    return best or ','


def _header(zf, member, sep):
    with zf.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding='utf-8-sig', errors='replace', newline='')
        return list(pd.read_csv(text, nrows=0, sep=sep).columns)


def _resolve(headers):
    norm = {}
    for h in headers:
        norm[_norm(h)] = h
    res = {}
    for key, cands in CSV_SPEC.items():
        hit = None
        for cand in cands:
            if cand in norm:
                hit = norm[cand]
                break
        res[key] = hit
    missing = [k for k, v in res.items() if v is None]
    return res, missing


def _pick_member(zf, path, checks):
    names = [n for n in zf.namelist() if n.lower().endswith(('.csv', '.txt', '.tsv'))]
    observed = {}
    for name in names:
        try:
            sep = _sniff_sep(zf, name)
            headers = _header(zf, name, sep)
        except Exception as exc:
            observed[name] = f'header read error {type(exc).__name__}'
            continue
        cols, missing = _resolve(headers)
        observed[name] = {'headers': headers, 'missing': missing}
        if not missing:
            return name, cols, sep
    detail = f'archive {path.name} members {names} observations {str(observed)[:2500]}'
    _add(
        checks,
        'required_t100_fields',
        f'no usable member in archive {path.name} members {names}',
        f'a member whose header resolves every required field {sorted(CSV_SPEC)}',
        False,
        detail=detail[:3800],
    )
    raise AssertionError('unreachable')


def _read_chunks(zf, member, usecols, sep):
    with zf.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding='utf-8-sig', errors='replace', newline='')
        reader = pd.read_csv(
            text,
            usecols=usecols,
            sep=sep,
            dtype=str,
            chunksize=250000,
            keep_default_na=True,
            na_values=['', 'NA', 'N/A', 'null', 'NULL'],
        )
        for chunk in reader:
            yield chunk


def _aggregate(chunk, cols, class_counter, counters):
    raw = chunk
    counters['total_rows'] += len(raw)
    cls = raw[cols['class']].astype('string').str.strip().str.upper()
    class_counter.update(cls.fillna('').value_counts().to_dict())
    d = pd.DataFrame({
        'carrier': raw[cols['carrier']].astype('string').str.strip(),
        'origin_market': pd.to_numeric(raw[cols['origin_market']], errors='coerce'),
        'dest_market': pd.to_numeric(raw[cols['dest_market']], errors='coerce'),
        'year': pd.to_numeric(raw[cols['year']], errors='coerce'),
        'month': pd.to_numeric(raw[cols['month']], errors='coerce'),
        'class': cls,
        'dep_raw': pd.to_numeric(raw[cols['dep']], errors='coerce'),
        'seats_raw': pd.to_numeric(raw[cols['seats']], errors='coerce'),
        'pax_raw': pd.to_numeric(raw[cols['passengers']], errors='coerce'),
        'distance': pd.to_numeric(raw[cols['distance']], errors='coerce'),
    })
    valid = (
        d['carrier'].notna()
        & d['origin_market'].notna()
        & d['dest_market'].notna()
        & d['year'].notna()
        & d['month'].notna()
        & (d['month'] >= 1)
        & (d['month'] <= 12)
    )
    counters['missing_key_rows'] += int((~valid).sum())
    d = d[valid].copy()
    d = d[(d['year'] >= 2010) & (d['year'] <= 2020)].copy()
    counters['used_rows'] += len(d)
    if d.empty:
        return pd.DataFrame()
    d['is_f'] = d['class'].eq('F')
    d['dep_missing'] = d['dep_raw'].isna().astype('int8')
    d['dep_known'] = d['dep_raw'].fillna(0.0)
    d['seats_missing'] = d['seats_raw'].isna().astype('int8')
    d['seats_known'] = d['seats_raw'].fillna(0.0)
    d['pax_missing'] = d['pax_raw'].isna().astype('int8')
    d['pax_known'] = d['pax_raw'].fillna(0.0)
    keys = ['carrier', 'origin_market', 'dest_market', 'year', 'month']
    all_agg = d.groupby(keys, observed=True).agg(
        dep_all_known=('dep_known', 'sum'),
        dep_all_missing=('dep_missing', 'sum'),
        rows_all=('dep_raw', 'size'),
    ).reset_index()
    f = d[d['is_f']]
    if len(f):
        f = f.copy()
        f['distance_positive'] = f['distance'].where(f['distance'] > 0)
        counters['distance_nonpositive_f_rows'] += int((f['distance'].notna() & (f['distance'] <= 0)).sum())
        counters['distance_missing_f_rows'] += int(f['distance'].isna().sum())
        f_agg = f.groupby(keys, observed=True).agg(
            dep_f_known=('dep_known', 'sum'),
            dep_f_missing=('dep_missing', 'sum'),
            seats_f_known=('seats_known', 'sum'),
            seats_f_missing=('seats_missing', 'sum'),
            pax_f_known=('pax_known', 'sum'),
            pax_f_missing=('pax_missing', 'sum'),
            rows_f=('dep_raw', 'size'),
        ).reset_index()
        fd = f[f['distance_positive'].notna()].groupby(keys + ['distance_positive'], observed=True).size().reset_index(name='n')
        if len(fd):
            fd = fd.sort_values(keys + ['n', 'distance_positive'], ascending=[True] * len(keys) + [True, True])
            fd_mode = fd.drop_duplicates(keys, keep='last')[keys + ['distance_positive']].rename(columns={'distance_positive': 'distance_mode'})
        else:
            fd_mode = pd.DataFrame(columns=keys + ['distance_mode'])
        f_agg = f_agg.merge(fd_mode, on=keys, how='left')
    else:
        f_agg = pd.DataFrame(columns=keys + [
            'dep_f_known', 'dep_f_missing', 'seats_f_known', 'seats_f_missing',
            'pax_f_known', 'pax_f_missing', 'rows_f', 'distance_mode',
        ])
    pair = all_agg.merge(f_agg, on=keys, how='outer')
    for c in ['dep_all_known', 'dep_all_missing', 'rows_all', 'dep_f_known', 'dep_f_missing', 'rows_f']:
        if c not in pair.columns:
            pair[c] = 0
    for c in ['seats_f_known', 'seats_f_missing', 'pax_f_known', 'pax_f_missing', 'distance_mode']:
        if c not in pair.columns:
            pair[c] = np.nan
    pair['dep_all_known'] = pd.to_numeric(pair['dep_all_known'], errors='coerce').fillna(0).astype('int64')
    pair['dep_all_missing'] = pd.to_numeric(pair['dep_all_missing'], errors='coerce').fillna(0).astype('int64')
    pair['rows_all'] = pd.to_numeric(pair['rows_all'], errors='coerce').fillna(0).astype('int64')
    pair['dep_f_known'] = pd.to_numeric(pair['dep_f_known'], errors='coerce').fillna(0.0).astype('float64')
    pair['dep_f_missing'] = pd.to_numeric(pair['dep_f_missing'], errors='coerce').fillna(0).astype('int64')
    pair['rows_f'] = pd.to_numeric(pair['rows_f'], errors='coerce').fillna(0).astype('int64')
    pair['seats_f_missing'] = pd.to_numeric(pair['seats_f_missing'], errors='coerce').fillna(0).astype('int64')
    pair['pax_f_missing'] = pd.to_numeric(pair['pax_f_missing'], errors='coerce').fillna(0).astype('int64')
    return pair


def _build_panel(cache, checks):
    session = _new_session()
    state = _read_form(session)
    hidden_names = sorted(state['hidden'])[:12]
    select_names = sorted(state['selects'])[:12]
    _add(
        checks,
        'download_form_state',
        f'hidden_inputs {hidden_names} select_names {select_names}',
        'the ASP.NET download form exposes hidden viewstate inputs plus year and period selectors',
        bool(state['hidden']) and _select_name(state['selects'], ('cboyear', 'year')) is not None,
    )
    fields = _resolve_form_fields(state['boxes'], checks)
    state['fields'] = fields
    year_sel = _select_name(state['selects'], ('cboyear', 'year'))
    period_sel = _select_name(state['selects'], ('cboperiod', 'cboquarter', 'period'))
    geo_sel = _select_name(state['selects'], ('cbogeography', 'geography'))
    if year_sel is None or period_sel is None:
        names = sorted(state['selects'])
        raise AssertionError(('Download form is missing a year or period selector. Select names observed: ' + repr(names))[:4000])
    state['year_field'] = year_sel
    state['period_field'] = period_sel
    state['geo_field'] = geo_sel
    state['geo_value'] = 'All'
    if geo_sel is not None:
        for opt in state['selects'][geo_sel]:
            if _norm(opt) == 'all':
                state['geo_value'] = str(opt)
                break
    year_opts = [str(x) for x in state['selects'][year_sel]]
    missing_years = [str(y) for y in PANEL_YEARS if str(y) not in year_opts]
    _add(
        checks,
        'form_year_options',
        f'count {len(year_opts)} first {year_opts[:3]} last {year_opts[-3:]}',
        'the download form offers every panel year from 2010 through 2020',
        not missing_years,
        detail=f'missing years {missing_years}',
    )
    period_opts = [str(x) for x in state['selects'][period_sel]]
    all_value = None
    for opt in period_opts:
        if _norm(opt) == 'all':
            all_value = opt
            break
    state['all_value'] = all_value
    month_like = [o for o in period_opts if o.strip().isdigit()]
    _add(
        checks,
        'form_period_options',
        f'all_value {all_value} month_options {month_like[:14]}',
        'the period selector offers month values and may offer an All value',
        len(month_like) >= 2,
        detail=f'period options observed {period_opts[:30]}',
    )
    prov = 'not attempted'
    try:
        cached = fetch(PAGE, cache, 't100_download_page.html')
        prov = f'landing page cached as {Path(cached).name} bytes {Path(cached).stat().st_size}'
    except Exception as exc:
        prov = f'landing page fetch failed {type(exc).__name__}'
    archives = []
    for year in PANEL_YEARS:
        archives.extend(_download_year(session, state, year, cache))
    _add(
        checks,
        'source_archives',
        f'archive_count {len(archives)} sample {[Path(p).name for p in archives[:3]]} provenance {prov}',
        'publisher archives covering every month 2010-01 through 2020-02 are present as zip files',
        len(archives) > 0 and all(_is_zip(p) for p in archives),
    )
    parts = []
    class_counter = Counter()
    counters = {
        'total_rows': 0,
        'used_rows': 0,
        'missing_key_rows': 0,
        'distance_nonpositive_f_rows': 0,
        'distance_missing_f_rows': 0,
    }
    for path in archives:
        with zipfile.ZipFile(path) as zf:
            member, cols, sep = _pick_member(zf, path, checks)
            keep = list(dict.fromkeys(cols.values()))
            for chunk in _read_chunks(zf, member, keep, sep):
                if not len(chunk):
                    continue
                part = _aggregate(chunk, cols, class_counter, counters)
                if len(part):
                    parts.append(part)
    if not parts:
        raise AssertionError('No T-100 segment rows parsed from the downloaded archives')
    panel = pd.concat(parts, ignore_index=True)
    keys = ['carrier', 'origin_market', 'dest_market', 'year', 'month']
    agg = {
        'dep_all_known': 'sum',
        'dep_all_missing': 'sum',
        'rows_all': 'sum',
        'dep_f_known': 'sum',
        'dep_f_missing': 'sum',
        'seats_f_known': 'sum',
        'seats_f_missing': 'sum',
        'pax_f_known': 'sum',
        'pax_f_missing': 'sum',
        'rows_f': 'sum',
        'distance_mode': 'first',
    }
    panel = panel.groupby(keys, as_index=False, observed=True).agg(agg)
    panel['year'] = panel['year'].astype('int64')
    panel['month'] = panel['month'].astype('int64')
    panel['origin_market'] = panel['origin_market'].astype('int64')
    panel['dest_market'] = panel['dest_market'].astype('int64')
    panel['ym'] = panel['year'] * 12 + panel['month'] - 1
    panel = panel[panel['ym'] <= PANEL_END_YM].copy()
    dups = int(panel.duplicated(keys).sum())
    _add(checks, 'pair_month_unique', f'duplicates {dups}', 'one carrier directed city market pair per month in the collapsed panel', dups == 0)
    expected = set(range(_ym(2010, 1), _ym(2020, 2) + 1))
    present = set(int(x) for x in panel['ym'].unique())
    missing_months = sorted(expected - present)
    _add(
        checks,
        'panel_month_coverage',
        f'present {len(present)} missing {missing_months[:12]}',
        'every month from 2010-01 through 2020-02 has carrier pair month rows',
        len(missing_months) == 0,
        detail=f'missing months {missing_months[:24]}',
    )
    min_dep_all = float(panel['dep_all_known'].min())
    _add(
        checks,
        'departures_nonnegative',
        f'min pair month departures {min_dep_all}',
        'DepPerformed is a nonnegative count of takeoffs made at an airport',
        min_dep_all >= 0,
        kind='unit',
    )
    return panel, class_counter, counters, archives, prov


def _candidate_frame(panel):
    cand = panel[
        (panel['dep_f_known'] >= 8)
        & (panel['pax_f_known'] > 0)
        & (panel['dep_f_missing'] == 0)
        & (panel['pax_f_missing'] == 0)
    ].copy()
    cand['row_id'] = np.arange(len(cand), dtype=np.int64)
    cand['split'] = cand['ym'].map(_split_for_ym)
    return cand[cand['split'].notna()].copy()


def _filter_prior(cand, panel):
    if len(cand) == 0:
        return cand, 0
    offs = pd.DataFrame({'off': np.arange(-12, 0, dtype=np.int64)})
    w = cand[['row_id', 'carrier', 'origin_market', 'dest_market', 'ym']].merge(offs, how='cross')
    w['wym'] = w['ym'] + w['off']
    active = panel[(panel['dep_all_known'] > 0) | (panel['dep_all_missing'] > 0)][
        ['carrier', 'origin_market', 'dest_market', 'ym']
    ]
    hit = w.merge(
        active,
        left_on=['carrier', 'origin_market', 'dest_market', 'wym'],
        right_on=['carrier', 'origin_market', 'dest_market', 'ym'],
        how='inner',
        suffixes=('', '_active'),
    )
    bad_ids = set(hit['row_id'].tolist())
    return cand[~cand['row_id'].isin(bad_ids)].copy(), len(bad_ids)


def _add_label(cand, panel):
    offs = pd.DataFrame({'off': [12, 13, 14]})
    w = cand[['row_id', 'carrier', 'origin_market', 'dest_market', 'ym']].merge(offs, how='cross')
    w['wym'] = w['ym'] + w['off']
    m = w.merge(
        panel[['carrier', 'origin_market', 'dest_market', 'ym', 'dep_all_known', 'dep_all_missing']],
        left_on=['carrier', 'origin_market', 'dest_market', 'wym'],
        right_on=['carrier', 'origin_market', 'dest_market', 'ym'],
        how='left',
        suffixes=('', '_panel'),
    )
    m['dep_all_known'] = m['dep_all_known'].fillna(0.0)
    m['dep_all_missing'] = m['dep_all_missing'].fillna(0).astype('int64')
    agg = m.groupby('row_id', as_index=False).agg(
        anniv_dep=('dep_all_known', 'sum'),
        anniv_missing=('dep_all_missing', 'sum'),
    )
    out = cand.merge(agg, on='row_id', how='left')
    out['anniv_missing'] = out['anniv_missing'].fillna(0).astype('int64')
    unknown = int((out['anniv_missing'] > 0).sum())
    out = out[out['anniv_missing'] == 0].copy()
    out['discontinued'] = (out['anniv_dep'] == 0).astype(int)
    return out, unknown


def _add_pair3(cand, panel):
    offs = pd.DataFrame({'off': [-3, -2, -1]})
    w = cand[['row_id', 'carrier', 'origin_market', 'dest_market', 'ym']].merge(offs, how='cross')
    w['wym'] = w['ym'] + w['off']
    p = panel[['carrier', 'origin_market', 'dest_market', 'ym', 'dep_all_known', 'dep_all_missing']].rename(
        columns={'carrier': 'other_carrier', 'ym': 'pym'}
    )
    m = w.merge(
        p,
        left_on=['origin_market', 'dest_market', 'wym'],
        right_on=['origin_market', 'dest_market', 'pym'],
        how='left',
    )
    m['dep_all_known'] = m['dep_all_known'].fillna(0.0)
    m['dep_all_missing'] = m['dep_all_missing'].fillna(0).astype('int64')
    agg = m.groupby('row_id', as_index=False).agg(
        pair_dep=('dep_all_known', 'sum'),
        pair_miss=('dep_all_missing', 'sum'),
    )
    comp = m[
        (m['dep_all_known'] > 0)
        & m['other_carrier'].notna()
        & (m['other_carrier'] != m['carrier'])
    ].groupby('row_id')['other_carrier'].nunique().rename('competing').reset_index()
    out = cand.merge(agg, on='row_id', how='left').merge(comp, on='row_id', how='left')
    out['competing'] = out['competing'].fillna(0)
    out.loc[out['pair_miss'] > 0, 'pair_dep'] = np.nan
    out.loc[out['pair_miss'] > 0, 'competing'] = np.nan
    return out


def _add_panel_features(cand, panel):
    cm = panel.groupby(['carrier', 'ym'], observed=True).agg(
        dep=('dep_all_known', 'sum'), miss=('dep_all_missing', 'sum')
    ).reset_index()
    cm_dep = {(r.carrier, int(r.ym)): float(r.dep) for r in cm.itertuples(index=False)}
    cm_miss = {(r.carrier, int(r.ym)): int(r.miss) for r in cm.itertuples(index=False)}
    om = panel.groupby(['origin_market', 'ym'], observed=True).agg(
        dep=('dep_all_known', 'sum'), miss=('dep_all_missing', 'sum')
    ).reset_index()
    om_dep = {(int(r.origin_market), int(r.ym)): float(r.dep) for r in om.itertuples(index=False)}
    om_miss = {(int(r.origin_market), int(r.ym)): int(r.miss) for r in om.itertuples(index=False)}
    dm = panel.groupby(['dest_market', 'ym'], observed=True).agg(
        dep=('dep_all_known', 'sum'), miss=('dep_all_missing', 'sum')
    ).reset_index()
    dm_dep = {(int(r.dest_market), int(r.ym)): float(r.dep) for r in dm.itertuples(index=False)}
    dm_miss = {(int(r.dest_market), int(r.ym)): int(r.miss) for r in dm.itertuples(index=False)}
    active = panel[panel['dep_all_known'] > 0][['carrier', 'origin_market', 'dest_market', 'ym']].drop_duplicates()
    carrier_pairs = {}
    for (c, ym), grp in active.groupby(['carrier', 'ym'], observed=True):
        carrier_pairs[(c, int(ym))] = set(zip(grp['origin_market'].astype(int), grp['dest_market'].astype(int)))
    co = panel[panel['dep_all_known'] > 0].groupby(['carrier', 'origin_market'], observed=True)['ym'].apply(
        lambda s: set(int(x) for x in s)
    ).to_dict()
    co_miss = panel[panel['dep_all_missing'] > 0].groupby(['carrier', 'origin_market'], observed=True)['ym'].apply(
        lambda s: set(int(x) for x in s)
    ).to_dict()
    cd = panel[panel['dep_all_known'] > 0].groupby(['carrier', 'dest_market'], observed=True)['ym'].apply(
        lambda s: set(int(x) for x in s)
    ).to_dict()
    cd_miss = panel[panel['dep_all_missing'] > 0].groupby(['carrier', 'dest_market'], observed=True)['ym'].apply(
        lambda s: set(int(x) for x in s)
    ).to_dict()
    c12 = {}

    def carrier12(c, ym):
        key = (c, int(ym))
        if key in c12:
            return c12[key]
        dep = 0.0
        miss = 0
        for k in range(int(ym) - 11, int(ym) + 1):
            dep += cm_dep.get((c, k), 0.0)
            miss += cm_miss.get((c, k), 0)
        c12[key] = (dep, miss)
        return c12[key]

    ca12 = {}

    def carrier_active12(c, ym):
        key = (c, int(ym))
        if key in ca12:
            return ca12[key]
        pairs = set()
        miss = 0
        for k in range(int(ym) - 11, int(ym) + 1):
            pairs |= carrier_pairs.get((c, k), set())
            miss += cm_miss.get((c, k), 0)
        ca12[key] = np.nan if miss > 0 else float(len(pairs))
        return ca12[key]

    at = {}

    def attrition(c, ym):
        key = (c, int(ym))
        if key in at:
            return at[key]
        prev = carrier_pairs.get((c, int(ym) - 12), set())
        now = carrier_pairs.get((c, int(ym)), set())
        miss = cm_miss.get((c, int(ym) - 12), 0) + cm_miss.get((c, int(ym)), 0)
        at[key] = np.nan if (miss > 0 or len(prev) == 0) else float(len(prev - now)) / float(len(prev))
        return at[key]

    m12 = {}

    def market12(kind, m, ym):
        key = (kind, int(m), int(ym))
        if key in m12:
            return m12[key]
        dmap, mmap = (om_dep, om_miss) if kind == 'o' else (dm_dep, dm_miss)
        dep = 0.0
        miss = 0
        for k in range(int(ym) - 11, int(ym) + 1):
            dep += dmap.get((int(m), k), 0.0)
            miss += mmap.get((int(m), k), 0)
        m12[key] = np.nan if miss > 0 else dep
        return m12[key]

    def new_at(maps, misses, c, m, ym):
        act = maps.get((c, int(m)), set())
        mis = misses.get((c, int(m)), set())
        rng = range(int(ym) - 12, int(ym))
        if any(k in mis for k in rng):
            return np.nan
        return 1.0 if not any(k in act for k in rng) else 0.0

    rows = []
    for r in cand.itertuples(index=False):
        c = r.carrier
        o = int(r.origin_market)
        d = int(r.dest_market)
        ym = int(r.ym)
        y = int(r.year)
        mth = int(r.month)
        dep12, miss12 = carrier12(c, ym)
        seats = r.seats_f_known
        pax = r.pax_f_known
        spd = np.nan
        if r.dep_f_known > 0 and r.seats_f_missing == 0 and not pd.isna(seats):
            spd = float(seats) / float(r.dep_f_known)
        lf = np.nan
        if (
            r.seats_f_missing == 0
            and r.pax_f_missing == 0
            and not pd.isna(seats)
            and not pd.isna(pax)
            and float(seats) > 0
        ):
            lf = float(pax) / float(seats)
        dist = np.nan
        if not pd.isna(r.distance_mode) and float(r.distance_mode) > 0:
            dist = float(r.distance_mode)
        rows.append({
            'episode_id': f'{c}-{o}-{d}-{y:04d}{mth:02d}',
            'entry_month': f'{y:04d}-{mth:02d}-01',
            'unique_carrier': c,
            'origin_city_market_id': o,
            'dest_city_market_id': d,
            'distance_miles': dist,
            'entry_calendar_month': mth,
            'carrier_departures_prior_12m': np.nan if miss12 > 0 else dep12,
            'carrier_active_pairs_prior_12m': carrier_active12(c, ym),
            'carrier_route_attrition_prior_12m': attrition(c, ym),
            'origin_market_departures_prior_12m': market12('o', o, ym),
            'dest_market_departures_prior_12m': market12('d', d, ym),
            'competing_carriers_prior_3m': r.competing,
            'pair_departures_prior_3m': r.pair_dep,
            'entry_month_departures': float(r.dep_f_known),
            'entry_month_seats_per_departure': spd,
            'entry_month_load_factor': lf,
            'carrier_new_at_origin': new_at(co, co_miss, c, o, ym),
            'carrier_new_at_dest': new_at(cd, cd_miss, c, d, ym),
            'discontinued': int(r.discontinued),
            'split': _split_for_ym(ym),
        })
    return pd.DataFrame(rows)


def _build_rows(panel, checks):
    cand = _candidate_frame(panel)
    _add(
        checks,
        'entry_candidates_before_prior',
        f'rows {len(cand)}',
        'at least one Class F entry month pair row with performed departures and passengers inside a frozen window',
        len(cand) > 0,
    )
    pax = pd.to_numeric(cand['pax_f_known'], errors='coerce')
    _add(
        checks,
        'entry_passengers_positive',
        f'min {float(pax.min())} nulls {int(pax.isna().sum())} rows {len(pax)}',
        'every candidate entry month carries Class F passengers greater than zero',
        float(pax.min()) > 0 and int(pax.isna().sum()) == 0,
    )
    before = len(cand)
    cand, bad = _filter_prior(cand, panel)
    _, remaining_bad = _filter_prior(cand, panel)
    _add(
        checks,
        'prior_pair_zero',
        f'before {before} excluded {bad} after {len(cand)} remaining_bad {remaining_bad}',
        'every included entry has zero performed departures on the directed pair in months M-12 through M-1',
        remaining_bad == 0 and len(cand) > 0,
        detail=f'rows failing the prior zero departure rule after filtering {remaining_bad}',
    )
    cand, unknown = _add_label(cand, panel)
    remaining_unknown = int((cand['anniv_missing'] > 0).sum())
    _add(
        checks,
        'label_unknown_excluded',
        f'excluded_unknown {unknown} remaining_unknown {remaining_unknown} labelled {len(cand)}',
        'rows whose anniversary window has missing performed departures are excluded and never labelled negative',
        remaining_unknown == 0,
    )
    if len(cand) == 0:
        raise AssertionError('No eligible T-100 entry rows after the population and label filters')
    cand = _add_pair3(cand, panel)
    rows = _add_panel_features(cand, panel)
    _add(
        checks,
        'entry_population',
        f'rows {len(rows)}',
        'at least one eligible entry episode with a complete anniversary label',
        len(rows) > 0,
    )
    return rows


def _finalize(rows, checks, counters):
    rows = rows.sort_values([TIME_COL, ID_COL], kind='mergesort').reset_index(drop=True)
    bad_target = int((~rows[TARGET].isin([0, 1])).sum())
    null_target = int(rows[TARGET].isna().sum())
    _add(
        checks,
        'target_binary',
        f'bad {bad_target} null {null_target}',
        'the discontinued target is 0 or 1 with no missing values',
        bad_target == 0 and null_target == 0,
    )
    dups = int(rows[ID_COL].duplicated().sum())
    _add(checks, 'stable_ids_unique', f'duplicates {dups}', 'episode_id is unique across all splits', dups == 0)
    counts = {f: int(rows[f].notna().sum()) for f in FEATURES}
    empty = [f for f, c in counts.items() if c == 0]
    _add(
        checks,
        'declared_features_nonempty',
        f'non_null_counts {counts}',
        'every declared feature has at least one non null value',
        not empty,
        detail=f'features with no values {empty}',
    )
    entry_dep = pd.to_numeric(rows['entry_month_departures'], errors='coerce')
    nonint = int((np.abs(entry_dep - np.round(entry_dep)) > 1e-9).sum())
    _add(
        checks,
        'entry_threshold_unit',
        f'min {float(entry_dep.min())} non_integer {nonint}',
        'DepPerformed is an integer takeoff count and the frozen population requires at least 8 Class F departures in month M',
        float(entry_dep.min()) >= 8 and nonint == 0,
        kind='unit',
    )
    dist = pd.to_numeric(rows['distance_miles'], errors='coerce').dropna()
    nonpos_dist = int((dist <= 0).sum())
    source_distance_nonpositive = int(counters.get('distance_nonpositive_f_rows', 0))
    source_distance_missing = int(counters.get('distance_missing_f_rows', 0))
    _add(
        checks,
        'distance_miles_positive',
        f'shipped nonpositive {nonpos_dist} nonnull {len(dist)} source_class_F_nonpositive {source_distance_nonpositive} source_class_F_missing {source_distance_missing}',
        'the reported nonstop segment distance is positive statute miles; nonpositive recorded source values are excluded from the modal distance and shipped as missing, never as zero or negative',
        nonpos_dist == 0 and len(dist) > 0,
        kind='unit',
    )
    spd = pd.to_numeric(rows['entry_month_seats_per_departure'], errors='coerce').dropna()
    nonpos_spd = int((spd <= 0).sum())
    _add(
        checks,
        'seats_per_departure_positive',
        f'nonpositive {nonpos_spd} nonnull {len(spd)}',
        'seats per performed departure is a positive capacity ratio',
        nonpos_spd == 0 and len(spd) > 0,
        kind='unit',
    )
    lf = pd.to_numeric(rows['entry_month_load_factor'], errors='coerce').dropna()
    finite_lf = bool(np.isfinite(lf).all()) if len(lf) else False
    neg_lf = int((lf < 0).sum())
    _add(
        checks,
        'load_factor_finite',
        f'nonnull {len(lf)} finite {finite_lf} negative {neg_lf}',
        'load factor is a finite reported passenger to seat ratio and is not clipped',
        finite_lf and neg_lf == 0,
        kind='unit',
    )
    by = {s: rows[rows['split'] == s].copy() for s in ['train', 'eval', 'holdout']}
    split_counts = {s: len(v) for s, v in by.items()}
    _add(checks, 'splits_nonempty', f'counts {split_counts}', 'all three frozen splits contain rows', all(v > 0 for v in split_counts.values()))
    for s, df in by.items():
        start, end = WINDOWS[s]
        mn = df[TIME_COL].min()
        mx = df[TIME_COL].max()
        ok = (start is None or mn >= start) and (end is None or mx < end)
        _add(checks, f'split_window_{s}', f'min {mn} max {mx}', f'half open window from {start} to {end}', ok)
    order = ['train', 'eval', 'holdout']
    for a, b in zip(order, order[1:]):
        a_end = pd.to_datetime(by[a][TIME_COL].max())
        b_start = pd.to_datetime(by[b][TIME_COL].min())
        gap = (b_start.year - a_end.year) * 12 + (b_start.month - a_end.month)
        _add(
            checks,
            f'label_window_embargo_{a}_{b}',
            f'last {a} entry month {a_end.date()} first {b} entry month {b_start.date()} gap months {gap}',
            f'at least 15 months separate the last {a} entry month from the first {b} entry month so the 14 month label window cannot overlap the next split',
            gap >= 15,
        )
    pos = int((by['train'][TARGET] == 1).sum())
    neg = int((by['train'][TARGET] == 0).sum())
    _add(checks, 'train_target_balance', f'positives {pos} negatives {neg}', 'the training split contains both outcome classes', pos > 0 and neg > 0)
    return by


def _write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(path, index=False)


def build(output, cache):
    output = Path(output)
    cache = Path(cache)
    (output / 'public').mkdir(parents=True, exist_ok=True)
    (output / 'private').mkdir(parents=True, exist_ok=True)
    checks = []
    panel, class_counter, counters, archives, prov = _build_panel(cache, checks)
    f_count = class_counter.get('F', 0)
    non_f = sum(v for k, v in class_counter.items() if k != 'F')
    top = class_counter.most_common(12)
    _add(
        checks,
        'class_F_present',
        f'F rows {f_count} top classes {top}',
        'service class F scheduled passenger rows are present in the T-100 segment download',
        f_count > 0,
    )
    total_rows = counters['total_rows']
    used_rows = counters['used_rows']
    missing_keys = counters['missing_key_rows']
    _add(
        checks,
        'non_F_classes_counted',
        f'non_F rows {non_f} total rows {total_rows} top classes {top}',
        'other service classes are counted and are not required to be absent from the publisher download',
        non_f > 0 and total_rows > 0,
    )
    _add(
        checks,
        'key_completeness',
        f'total rows {total_rows} used rows {used_rows} missing key rows {missing_keys}',
        'rows with carrier and city market and year and month keys are used while rows missing keys are counted',
        used_rows > 0 and missing_keys < total_rows,
    )
    rows = _build_rows(panel, checks)
    by = _finalize(rows, checks, counters)
    _write(by['train'], output / 'public' / 'train.csv')
    _write(by['eval'], output / 'public' / 'eval.csv')
    _write(by['holdout'], output / 'private' / 'holdout.csv')
    return checks
