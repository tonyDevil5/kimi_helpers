import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const csvPath = path.resolve(here, "../tmp/fed_hike_target_bbg.csv");
const templatePath = path.join(here, "index.template.html");
const outputPath = path.join(here, "dist/index.html");

const lines = fs.readFileSync(csvPath, "utf8").trim().split(/\r?\n/);
const col = Object.fromEntries(lines[0].split(",").map((name, i) => [name, i]));
const num = (parts, name) => {
  const value = parts[col[name]];
  return value === "" || value == null ? NaN : +value;
};
const rows = lines.slice(1).map((line) => {
  const parts = line.split(",");
  return {
    date: parts[col.date],
    fnccBasis: num(parts, "fncc_basis"),
    fed: num(parts, "target_rate"),
    ten: num(parts, "ust_10y"),
    twosTens: num(parts, "twos_tens"),
    fivesThirties: num(parts, "fives_thirties"),
    nmcmfr30: num(parts, "nmcmfr30"),
    creditOas: num(parts, "credit_oas"),
    spx: num(parts, "spx"),
    oil: num(parts, "oil"),
    gold: num(parts, "gold"),
    cpi: num(parts, "cpi_yoy"),
    urate: num(parts, "urate"),
    fedObserved: num(parts, "target_rate_observed"),
    tenObserved: num(parts, "ust_10y_observed"),
    twosTensObserved: num(parts, "twos_tens_observed"),
    fivesThirtiesObserved: num(parts, "fives_thirties_observed"),
    fnccObserved: num(parts, "fncc_basis_observed"),
    nmcmfr30Observed: num(parts, "nmcmfr30_observed"),
    creditOasObserved: num(parts, "credit_oas_observed"),
    spxObserved: num(parts, "spx_observed"),
    oilObserved: num(parts, "oil_observed"),
    goldObserved: num(parts, "gold_observed"),
    cpiObserved: num(parts, "cpi_yoy_observed"),
    urateObserved: num(parts, "urate_observed"),
  };
});
const byDate = new Map(rows.map((row) => [row.date, row]));
const latest = rows.at(-1).date;
const eventDefs = [
  ["1999-06-30", "06/30/99", "1999"],
  ["2004-06-30", "06/30/04", "2004"],
  ["2016-12-14", "12/14/16", "2016"],
  ["2022-03-16", "03/16/22", "2022"],
  ["2026-09-16", "09/16/26", "2026"],
];
const iso = (date) => date.toISOString().slice(0, 10);
const addDays = (date, days) => new Date(date.getTime() + days * 86400000);
const round = (value, places = 3) => value == null ? null : +value.toFixed(places);

const events = eventDefs.map(([eventDate, label, short]) => {
  const start = new Date(`${eventDate}T12:00:00Z`);
  const day0 = byDate.get(eventDate);
  const series = [];
  for (let day = -60; day <= 500; day += 1) {
    const date = iso(addDays(start, day));
    const row = byDate.get(date);
    series.push({
      day,
      date,
      fed: row ? round(row.fed, 2) : null,
      ten: row ? round(row.ten, 3) : null,
      twosTens: row ? round(row.twosTens, 1) : null,
      fivesThirties: row ? round(row.fivesThirties, 1) : null,
      fnccBasis: row ? round(row.fnccBasis, 1) : null,
      mort: row ? round(row.nmcmfr30, 3) : null,
      creditOas: row ? round(row.creditOas, 1) : null,
      spx: row ? round(row.spx, 2) : null,
      oil: row ? round(row.oil, 2) : null,
      gold: row ? round(row.gold, 2) : null,
      cpi: row ? round(row.cpi, 2) : null,
      urate: row ? round(row.urate, 2) : null,
      twosTensChange: row ? round(row.twosTens - day0.twosTens, 1) : null,
      fivesThirtiesChange: row ? round(row.fivesThirties - day0.fivesThirties, 1) : null,
      fnccChange: row ? round(row.fnccBasis - day0.fnccBasis, 1) : null,
      fedChange: row ? round((row.fed - day0.fed) * 100, 1) : null,
      tenChange: row ? round((row.ten - day0.ten) * 100, 1) : null,
      mortChange: row ? round((row.nmcmfr30 - day0.nmcmfr30) * 100, 1) : null,
      creditOasChange: row ? round(row.creditOas - day0.creditOas, 1) : null,
      spxChange: row ? round((row.spx / day0.spx - 1) * 100, 1) : null,
      oilChange: row ? round((row.oil / day0.oil - 1) * 100, 1) : null,
      goldChange: row ? round((row.gold / day0.gold - 1) * 100, 1) : null,
      cpiChange: row ? round((row.cpi - day0.cpi) * 100, 1) : null,
      urateChange: row ? round((row.urate - day0.urate) * 100, 1) : null,
    });
  }
  const at = (day, key) => series.find((point) => point.day === day)?.[key] ?? null;
  const latestDay = Math.min(500, Math.floor((new Date(`${latest}T12:00:00Z`) - start) / 86400000));
  return {
    eventDate, label, short,
    partial: latestDay < 500,
    availableThrough: latestDay < 500 ? latest : iso(addDays(start, 500)),
    day0Fed: day0.fed,
    day0Ten: day0.ten,
    pre60Ten: round(at(0, "tenChange") - at(-60, "tenChange"), 1),
    day30Ten: at(30, "tenChange"),
    day60Ten: at(60, "tenChange"),
    day120Ten: at(120, "tenChange"),
    day250Ten: at(250, "tenChange"),
    day500Ten: at(500, "tenChange"),
    latestTen: at(latestDay, "tenChange"),
    series,
  };
});

const payload = { latest, events };
const template = fs.readFileSync(templatePath, "utf8");
fs.writeFileSync(outputPath, template.replace("/*__DATA__*/", JSON.stringify(payload)), "utf8");
console.log(outputPath);
