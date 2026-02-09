#!/usr/bin/env Rscript

if (!require("haven", quietly = TRUE)) {
  install.packages("haven", repos="http://cran.rstudio.com/", quiet=TRUE)
}
library(haven)

wvs_input <- "data/raw/sav/wvs_trend_1981-2022.sav"
wvs_csv <- "data/raw/csv/wvs_trend_1981-2022.csv"
wvs_labels <- "data/raw/csv/wvs_variable_labels.csv"

evs_input <- "data/raw/sav/evs_trend_1981-2017.sav"
evs_csv <- "data/raw/csv/evs_trend_1981-2017.csv"
evs_labels <- "data/raw/csv/evs_variable_labels.csv"

cat("Converting WVS...\n")
if (file.exists(wvs_input)) {
  wvs <- read_sav(wvs_input, encoding = "latin1")
  write.csv(wvs, wvs_csv, row.names = FALSE, fileEncoding = "UTF-8")

  wvs_label_list <- sapply(names(wvs), function(v) {
    l <- attr(wvs[[v]], "label")
    if (is.null(l)) "" else as.character(l)
  })
  write.csv(data.frame(variable = names(wvs), label = unname(wvs_label_list)),
            wvs_labels, row.names = FALSE, fileEncoding = "UTF-8")

  cat("  WVS:", nrow(wvs), "rows,", ncol(wvs), "columns\n")
  cat("  Saved:", wvs_csv, "\n")
} else {
  cat("  WVS file not found, skipping\n")
}

cat("\nConverting EVS...\n")
if (file.exists(evs_input)) {
    evs <- read_sav(evs_input, encoding = "latin1")
    write.csv(evs, evs_csv, row.names = FALSE, fileEncoding = "UTF-8")

  evs_label_list <- sapply(names(evs), function(v) {
    l <- attr(evs[[v]], "label")
    if (is.null(l)) "" else as.character(l)
  })
  write.csv(data.frame(variable = names(evs), label = unname(evs_label_list)),
            evs_labels, row.names = FALSE, fileEncoding = "UTF-8")

  cat("  EVS:", nrow(evs), "rows,", ncol(evs), "columns\n")
  cat("  Saved:", evs_csv, "\n")
} else {
  cat("  EVS file not found, skipping\n")
}

cat("\nDone\n")