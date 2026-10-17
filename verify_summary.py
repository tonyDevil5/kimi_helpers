import json
path = r'D:\python\kimi_helpers\dealer_summary.json'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print('File saved:', path)
print('Title:', data['title'])
print('Date range:', data['date_range'])
print()
for section, banks in data['sections'].items():
    print(f'【{section}】')
    if not banks:
        print('  （无）')
    for bank, summary in banks.items():
        if len(summary) > 80:
            print(f'  • {bank}: {summary[:80]}...')
        else:
            print(f'  • {bank}: {summary}')
    print()
