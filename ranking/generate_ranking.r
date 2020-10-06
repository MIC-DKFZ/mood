library(challengeR)

##### Sample

data=read.csv("sample_ranking.csv")
challenge=as.challenge(data,by="task",
                        algorithm="alg_name", case="case", value="value",smallBetter = FALSE
                        )
ranking=challenge%>%aggregateThenRank(FUN = mean, 
                                      na.treat=0,
                                      ties.method = "min" 
)
meanRanks=ranking%>%consensus(method = "euclidean")
ranking%>%report(consensus=meanRanks,
         title="MOOD 2020 - Sample",
         file = "sample_report",
         format = "PDF", # format can be "PDF", "HTML" or "Word"
         latex_engine="pdflatex"#LaTeX engine for producing PDF output. Options are "pdflatex", "lualatex", and "xelatex"
         )



##### Pixel

data=read.csv("pixel_ranking.csv")
challenge=as.challenge(data,by="task",
                        algorithm="alg_name", case="case", value="value",smallBetter = FALSE
                        )
ranking=challenge%>%aggregateThenRank(FUN = mean, 
                                      na.treat=0,
                                      ties.method = "min" 
)
meanRanks=ranking%>%consensus(method = "euclidean")
ranking%>%report(consensus=meanRanks,
         title="MOOD 2020 - Pixel",
         file = "pixel_report",
         format = "PDF", # format can be "PDF", "HTML" or "Word"
         latex_engine="pdflatex"#LaTeX engine for producing PDF output. Options are "pdflatex", "lualatex", and "xelatex"
         )

