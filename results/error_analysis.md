# Error analysis: distilled student

13 failures out of 50 (`results/eval/student_distilled_per_question.jsonl`), labelled by hand in `results/error_labels.yaml`. "Teacher too" = the teacher also fails this question.

## Answers a question the data can't answer, with an invented proxy (3)

- **q19**: What was the total profit in 2011?
  invents cost_price = 1 and filters on 1 January 2011. Expected `None`.
- **q20**: What is the average delivery time in days?
  'delivery time' becomes the gap between consecutive invoice dates. Expected `None`.
- **h10** (teacher too): Which product category sells best?
  searches the word 'category' in Description (then crashes 4 times). Expected `None`.

## Computes on the last line but leaves `result` holding an intermediate value (2)

- **h06** (teacher too): What is the total revenue (Quantity times Price) of the invoices that are not cancellations (the invoice number does not start with C)?
  `result = df[...]['Revenue'] = ...` then `result.sum()` on its own line. Expected `20375065.778`.
- **h14**: What is the total revenue (Quantity times Price) for Australia?
  `result` = the per-row products; `result.sum()` is never stored. Expected `166511.91999999998`.

## Ignores a constraint or the requested format (3)

- **q06**: Apart from the United Kingdom, which country has the highest total revenue (Quantity times Price)?
  forgets 'apart from the United Kingdom', returns United Kingdom. Expected `EIRE`.
- **q15** (teacher too): By how many percent did Germany's total revenue (Quantity times Price) change from 2010 to 2011? Give the percentage, e.g. 12.5 for +12.5 %.
  compares January with December instead of 2010 with 2011. Expected `5.897514507693269`.
- **q16**: On which day of the week (for example Monday) were the most distinct invoices created?
  returns 3 (day number) instead of 'Thursday', and counts rows, not distinct invoices. Expected `Thursday`.

## Wrong aggregation or a pandas operator bug (4)

- **q05**: Apart from the United Kingdom, which country has the most distinct customers?
  nunique() on the whole column (no group-by), then returns the max Customer ID. Expected `Germany`.
- **q12**: How many rows have no customer ID?
  `isna() | col == ''`: `|` binds before `==`, so the count is 0. Expected `235287`.
- **h26**: What is the average number of rows per invoice?
  mean Quantity per invoice instead of the number of rows per invoice. Expected `19.4832550160364`.
- **h27** (teacher too): What is the median total revenue per invoice (Quantity times Price summed per invoice)?
  mean per invoice instead of the sum per invoice, before the median. Expected `193.765`.

## Crashes and the retries repeat the broken code (1)

- **q13**: Which month had the highest total revenue (Quantity times Price)? Answer in YYYY-MM format.
  formats a Period with :04d; attempts 3 and 4 are the same broken code. Expected `2011-11`.

Failures shared with the teacher: 4 of 13 (q15, h06, h10, h27).
