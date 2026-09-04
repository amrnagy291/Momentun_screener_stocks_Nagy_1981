"""
Sanity tests for data_fetch.py's pure parsing/filtering logic. Uses small,
literal fixtures shaped exactly like the real NASDAQ Trader directory files
(pipe-delimited, header row, trailing "File Creation Time" line) instead of
hitting the network -- data_fetch.py's actual downloads (get_sp500_tickers,
get_all_us_tickers, download_price_history) need live internet access and
are NOT covered here, by the same design as the rest of this project's test
suite (see the README's "Testing without the internet" section).
Run with: python3 test_data_fetch.py
"""

import data_fetch as df


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, name


NASDAQ_FIXTURE = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAAP|Pacer Barings CLO Market Flex ETF|G|N|N|100|Y|N
AACG|ATA Creativity Global - American Depositary Shares, each representing two common shares|S|N|N|100|N|N
AACI|Armada Acquisition Corp. III - Class A Ordinary Share|G|N|N|100|N|N
AACIU|Armada Acquisition Corp. III - Units|G|N|N|100|N|N
AACIW|Armada Acquisition Corp. III - Warrant|G|N|N|100|N|N
ZZZT|Nasdaq Test Symbol|G|Y|N|100|N|N
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
File Creation Time: 0904202608:00PM"""

OTHER_FIXTURE = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
A|Agilent Technologies, Inc. Common Stock|N|A|N|100|N|A
AA|Alcoa Corporation Common Stock |N|AA|N|100|N|AA
AAA|Alternative Access First Priority CLO Bond ETF|P|AAA|Y|100|N|AAA
AAC|Ares Acquisition Corporation III Class A Ordinary Shares|N|AAC|N|100|N|AAC
AAC.U|Ares Acquisition Corporation III Units, each consisting of one Class A ordinary share and one-tenth of one redeemable warrant|N|AAC.U|N|100|N|AAC=
AAC.W|Ares Acquisition Corporation III Redeemable warrants, each whole warrant exercisable for one Class A ordinary share at an exercise price of $11.50|N|AAC.WS|N|100|N|AAC+
BRK.B|Berkshire Hathaway Inc. Common Stock|N|BRK.B|N|100|N|BRK.B
File Creation Time: 0904202608:00PM"""


def main():
    tickers = df._parse_symbol_directory(NASDAQ_FIXTURE, OTHER_FIXTURE)
    print(f"    parsed tickers: {tickers}")

    check("ETF is excluded (AAAP)", "AAAP" not in tickers)
    check("ETF is excluded (AAA)", "AAA" not in tickers)
    check("test issue is excluded (ZZZT)", "ZZZT" not in tickers)
    check("unit is excluded (AACIU)", "AACIU" not in tickers)
    check("warrant is excluded (AACIW)", "AACIW" not in tickers)
    check("unit is excluded (AAC-U / AAC.U)", "AAC-U" not in tickers and "AAC.U" not in tickers)
    check("warrant is excluded (AAC-W / AAC.W)", "AAC-W" not in tickers and "AAC.W" not in tickers)

    check("ordinary Nasdaq stock included (AACG)", "AACG" in tickers)
    check("ordinary Nasdaq stock included (AAPL)", "AAPL" in tickers)
    check("SPAC ordinary share (no unit/warrant marker) included (AAC)", "AAC" in tickers)
    check("ordinary NYSE stock included (A)", "A" in tickers)
    check("ordinary NYSE stock included (AA)", "AA" in tickers)
    check("share-class dot converted to dash (BRK.B -> BRK-B)", "BRK-B" in tickers and "BRK.B" not in tickers)

    check("no duplicates", len(tickers) == len(set(tickers)))
    check("result is sorted", tickers == sorted(tickers))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
