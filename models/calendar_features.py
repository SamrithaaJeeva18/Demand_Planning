"""calendar_features.py - one row per month of Sri Lanka calendar facts and special events, built from the `holidays` package + a small events table."""
import holidays
import pandas as pd

KINDS = ("tax_change", "special_holiday", "school_vacation")      # the kinds of rows allowed in the events table

# What every column means. Shown by SriLankaCalendar().guide().   (column, group, meaning, method, numeric example, known on the 1st of the month?)
GUIDE = [
    ("time_idx", "key", "Months since April 2022 (April 2022 = 0)", "counting", "September 2025 = 41", "yes"),
    ("days_in_month", "calendar", "Days in the month", "counting", "April 2025 = 30", "yes"),
    ("weekdays", "calendar", "Monday-Friday days", "counting", "April 2025 = 22", "yes"),
    ("holidays", "calendar", "Holidays on any day of the week", "lookup in the holidays package", "April 2025 = 4", "yes"),
    ("holidays_weekday", "calendar", "Holidays that fall Monday-Friday", "lookup + weekday test", "April 2025 = 2 (Mon 14th, Fri 18th)", "yes"),
    ("working_days", "calendar", "Weekdays left after weekday holidays", "subtraction: weekdays - holidays_weekday", "April 2025: 22 - 2 = 20;  April 2023: 20 - 4 = 16", "yes"),
    ("poya_days", "calendar", "Full-moon Poya holidays", "match on the holiday name", "April 2025 = 1 (the 12th)", "yes"),
    ("long_weekends", "calendar", "Breaks of 3+ non-working days that START in the month", "run-length detection over weekend + holiday days", "April 2025 = 2 (12-14th and 18-20th)", "yes"),
    ("ramadan_days", "moving", "Days of Ramadan inside the month", "window = the 29 days before Eid al-Fitr; count days overlapping the month", "March 2025 = 29;  March 2024 = 19, April 2024 = 10", "yes, +-1 day (moon sighting)"),
    ("eid_fitr", "moving", "1 if Eid al-Fitr is in the month", "0/1 flag", "March 2025 = 1 (31st);  April 2024 = 1 (11th)", "yes, +-1 day"),
    ("eid_adha", "moving", "1 if Eid al-Adha is in the month", "0/1 flag", "June 2025 = 1 (7th);  June 2024 = 1 (17th)", "yes, +-1 day"),
    ("diwali", "moving", "1 if Diwali is in the month", "0/1 flag", "October 2022 = 1 (24th);  November 2023 = 1 (12th)", "yes"),
    ("good_friday", "moving", "1 if Good Friday is in the month", "0/1 flag", "March 2024 = 1 (29th);  April 2025 = 1 (18th)", "yes"),
    ("avurudu", "fixed", "1 if the Sinhala and Tamil New Year holidays are in the month", "0/1 flag", "every April = 1", "yes"),
    ("christmas", "fixed", "1 if Christmas is in the month", "0/1 flag", "every December = 1", "yes"),
    ("vat_change_pp", "tax_events", "VAT change taking effect in the month, in percentage points (+ rise, - cut)", "signed size from the events table", "June 2022 = +4, September 2022 = +3, January 2024 = +3, all other months 0", "yes (a change effective on the 1st is public by then)"),
    ("vat_next_change_pp", "tax_events", "Size of the change coming NEXT month - only if it was announced before this month began", "lead feature (the next month's value moved back one month) + point-in-time rule", "December 2023 = +3 (announced mid-Nov); August 2022 = 0 (announced 31 Aug, too late); May 2022 = 0 (announcement date unknown)", "yes, by the rule"),
    ("vat_rate", "tax_level", "VAT rate in force (%)", "step function: start rate + running total of the changes", "Apr-May 2022 = 8, Jun-Aug 2022 = 12, Sep 2022-Dec 2023 = 15, from Jan 2024 = 18", "yes"),
    ("school_vacation_days", "school", "School vacation days in the month", "overlap of date ranges with the month", "April 2025 = 9 (12-20th);  March 2025 = 17 (15-31st)", "yes - but only 2025 is filled in"),
    ("special_holiday_days", "explain_only", "Extra holidays declared at short notice (public-sector, bank, bad weather)", "overlap of dates with the month", "June 2022 = 3", "NO - declared days before; use it to explain odd months, not to forecast"),
]


class SriLankaCalendar:
    """One row per month: working days, holiday counts, moving festivals, tax changes and other events from a small editable table."""

    GROUPS = {                               # columns that may be given to a forecasting model, grouped so each group can be tested alone
        "calendar": ["working_days", "holidays_weekday", "poya_days", "long_weekends"],
        "moving": ["ramadan_days", "eid_fitr", "eid_adha", "diwali", "good_friday"],
        "fixed": ["avurudu", "christmas"],
        "tax_events": ["vat_change_pp", "vat_next_change_pp"],
        "tax_level": ["vat_rate"],
        "school": ["school_vacation_days"],
    }                                        # special_holiday_days is NOT listed: it is only known after the fact

    def __init__(self, events=None, vat_start=8.0):
        # events: table with columns kind, start, end, name, value (tax: signed change in points), announced (date the public learned of it)
        # vat_start: the VAT rate (%) in force just before the first month. 8 for April 2022.
        ev = pd.DataFrame(columns=["kind", "start", "end", "name", "value", "announced"]) if events is None else events.copy()
        for col in ("value", "announced"):
            if col not in ev.columns:
                ev[col] = pd.NA
        ev["start"], ev["end"], ev["announced"] = pd.to_datetime(ev["start"]), pd.to_datetime(ev["end"]), pd.to_datetime(ev["announced"])
        ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
        assert set(ev["kind"]) <= set(KINDS), f"unknown kind in events: {set(ev['kind']) - set(KINDS)}"
        self.events, self.vat_start = ev, float(vat_start)

    def guide(self):
        # Responsibility: the column dictionary as a table - what each column is, the method used, a numeric example.
        return pd.DataFrame(GUIDE, columns=["column", "group", "meaning", "method", "example", "known on the 1st?"])

    def coverage(self):
        # Responsibility: event rows per kind and year. A 0 means "not filled in yet" - do not trust a column built from that kind for that year.
        ev = self.events.assign(year=self.events["start"].dt.year)
        table = ev.pivot_table(index="kind", columns="year", values="name", aggfunc="count", fill_value=0)
        return table.reindex(index=list(KINDS), columns=range(2022, 2027), fill_value=0)

    @staticmethod
    def _overlap(start, end, ms, me):
        # Days of the range [start, end] that fall inside the month [ms, me].   Example: Apr 12-20 vs April -> 9;  Mar 15-31 vs April -> 0
        return max(0, (min(end, me) - max(start, ms)).days + 1)

    def _long_weekend_starts(self, periods, hol):
        # Responsibility: mark the first day of every break of 3 or more non-working days (weekend or holiday).
        # Example: Sat 12 Apr (Poya), Sun 13, Mon 14 (New Year) are all off -> a 3-day break starting on the 12th. A plain Sat+Sun (2 days) is not marked.
        days = pd.date_range(periods[0].start_time.normalize() - pd.Timedelta(days=7), periods[-1].end_time.normalize() + pd.Timedelta(days=7))
        off = pd.Series([(d.dayofweek >= 5) or (d in hol) for d in days], index=days)
        run = (off != off.shift()).cumsum()                                   # a new number every time off/on switches
        return off & (run != run.shift()) & (off.groupby(run).transform("size") >= 3)

    def _tax_columns(self, periods):
        # Responsibility: the three tax columns, one value per month.
        # vat_change_pp = the signed change in the month it takes effect;  vat_rate = start rate + running total of changes.
        # vat_next_change_pp = next month's change, but ONLY if it was announced before this month began (what a forecaster really knew).
        tax = self.events[self.events["kind"] == "tax_change"]
        change = pd.Series(0.0, index=periods)
        next_known = pd.Series(0.0, index=periods)
        for _, r in tax.iterrows():
            p = r["start"].to_period("M")
            if p in change.index:
                change[p] += r["value"]
            before = p - 1
            if before in next_known.index and pd.notna(r["announced"]) and r["announced"] < before.start_time:
                next_known[before] += r["value"]
        return pd.DataFrame({"vat_change_pp": change, "vat_next_change_pp": next_known, "vat_rate": self.vat_start + change.cumsum()})

    def monthly(self, first="2022-04", last="2026-12"):
        # Responsibility: the feature table, one row per month; time_idx 0 = `first`. Every column is explained in guide().
        periods = pd.period_range(first, last, freq="M")
        hol = {pd.Timestamp(d): n for d, n in holidays.country_holidays("LK", years=range(periods[0].year - 1, periods[-1].year + 2), language="en_US").items()}
        long_start = self._long_weekend_starts(periods, hol)
        eids = [d for d, n in hol.items() if "Eid al-Fitr" in n]
        flags = {"eid_fitr": "Eid al-Fitr", "eid_adha": "Eid al-Adha", "diwali": "Diwali", "good_friday": "Good Friday",
                 "avurudu": "Sinhala and Tamil New Year", "christmas": "Christmas"}
        rows = []
        for p in periods:
            ms, me = p.start_time.normalize(), p.end_time.normalize()
            month_days = pd.date_range(ms, me)
            inside = {d: n for d, n in hol.items() if ms <= d <= me}
            weekday_holidays = [d for d in inside if d.dayofweek < 5]
            row = {"month": str(p), "time_idx": (p - periods[0]).n, "days_in_month": len(month_days),
                   "weekdays": int((month_days.dayofweek < 5).sum()), "holidays": len(inside), "holidays_weekday": len(weekday_holidays)}
            row["working_days"] = row["weekdays"] - row["holidays_weekday"]
            row["poya_days"] = sum("Full Moon Poya" in n and not n.startswith("Day Following") for n in inside.values())
            row["long_weekends"] = int(long_start[ms:me].sum())
            row["ramadan_days"] = sum(self._overlap(e - pd.Timedelta(days=29), e - pd.Timedelta(days=1), ms, me) for e in eids)
            for col, key in flags.items():
                row[col] = int(any(key in n for n in inside.values()))
            for kind, col in (("school_vacation", "school_vacation_days"), ("special_holiday", "special_holiday_days")):
                g = self.events[self.events["kind"] == kind]
                row[col] = sum(self._overlap(s, e, ms, me) for s, e in zip(g["start"], g["end"]))
            rows.append(row)
        df = pd.DataFrame(rows)
        tax = self._tax_columns(periods).reset_index(drop=True)
        return pd.concat([df, tax], axis=1)