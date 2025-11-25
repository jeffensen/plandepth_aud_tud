* Encoding: UTF-8.

*** replace with path to repository on your machine

* CD 'C:\Users\...[ADD PATH TO REPOSITORY]'.
* CD 'C:\Users\s7340493\Nextcloud\SESYN\_B09\SATPD2_AUD\plandepth_aud\fitting\2025-02-03_LPP_1P_fixed_and_discounting_global'.
CD 'C:\Users\jstef\Nextcloud\SESYN\_B09\SATPD2_AUD\plandepth_aud\fitting\2025-02-03_LPP_1P_fixed_and_discounting_global'.

*** helpers
* Macro function to add value labels to a column according to labels stored in another column.
DEFINE addValueLabelsFromLabelCol (indexCol      = !TOKENS(1)
                                                        /labelCol        = !TOKENS(1)
                                                        /indexColStr   = !TOKENS(1)
                                                        /datasetname = !TOKENS(1))
  
    * aggregate data to get only one line for every index/label pair.
    dataset activate !datasetname.
    dataset declare agg.
    aggregate out=agg /break !indexCol !labelCol /nn=n.
    dataset activate agg.
    
    * use data to create syntax file with the value label commands.
    string cat (a50).
    compute cat=concat('"', !labelCol, '"').
    write outfile="temp_myLabelsSyntax.sps" /"add value labels "+!indexColStr+" ", !indexCol, cat, ".".
    execute.
    
    * getting back to the original data and execute the syntax.
    dataset activate !datasetname.
    insert file="temp_myLabelsSyntax.sps".
    
    * close temp files.
    dataset close agg.
    erase file="temp_myLabelsSyntax.sps".
                                                            
!ENDDEFINE. 


*** load subject-level CSV data

PRESERVE.
SET DECIMAL DOT.

GET DATA  /TYPE=TXT
  /FILE="SAT_subjectLevel.csv"
  /ENCODING='UTF8'
  /DELIMITERS=";"
  /QUALIFIER="'"
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /DATATYPEMIN PERCENTAGE=95.0
  /VARIABLES=
  ID A5
  group AUTO
  group2 AUTO
  condition AUTO
  a0 F10.3
  SAT_RT F10.2
  MeanPD F10.3
  SAT_gain F10.1
  SAT_keys_condition F1
  SAT_Total_points F4
  SAT_PER F10.2
  group_id F2
  subject 4X
  model_beta F10.3
  model_theta F10.3
  model_k F10.3
  WAIC F10.3
  WAIC_lowNoise F10.3
  WAIC_highNoise F10.3
  pWAIC F10.3
  DIC F10.3
  NLL_sampled F10.3
  nll_1staction_mean 17X
  nll_1staction_hinoise_mean 17X
  nll_1staction_lonoise_mean 18X
  BIC_mean 18X
  BIC_hinoise_mean F10.3
  BIC_lonoise_mean F10.3
  pseudo_Rsquare_1staction_mean 19X
  pseudo_Rsquare_1staction_hinoise_mean 19X
  pseudo_Rsquare_1staction_lonoise_mean 18X
  WAIC_marginal F10.3
  WAIC_pointwise F10.3
  pWAIC_pointwise F10.3
  RAV_MaxCORR F3
  RAV_CORR F3
  RAV_ERR F3
  RAV_Missing F3
  RAV_TooFast F3
  RAV_PER F10.2
  RAV_ACC F10.2
  RAV_RT F10.2
  RAV_RT_SD F10.2
  RAV_RT_Instr F10.2
  RAV_RT_Task F10.2
  RAV_CORR_Instr F3
  scid_tud_sum F2
  scid_aud_sum F2
  ftnd_sum F2
  audit_score F2
  age F3
  gender F2
  qf_tab_3m_smokingDays F3
  qf_tab_3m_smokingDays_sumOrWeekly F3
  qf_tab_3m_meanDailyCigarettes F3
  qf_tab_lifetime_smoker F3
  qf_alc_3m_drinkingDays F3
  qf_alc_3m_drinkingDays_sumOrWeekly F3
  qf_alc_3m_bingingDays F3
  qf_alc_3m_bingingDays_sumOrWeekly F3
  qf_alc_amount_perDay F10.2
  qf_alc_amount_perWorkday F10.2
  qf_alc_amount_perWeekendDay F10.2
  qf_alc_amount_lastDay F10.2
  group_HC F2
  group_AUD_only F2
  group_TUD_only F2
  group_AUD_and_TUD F2
  sozio_income F2
  sozio_graduat F2
  sozio_graduat_other F2
  electronics_use F2
  games_use F2
  working F2
  sozio_job_status F2
  nfc_score F2
  bis15_sum_planning F2
  bis15_sum_motor F2
  bis15_sum_attention F2
  bis15_sum_all F2
  gender_label AUTO
  sozio_graduat_label AUTO
  HEEQ F2
  sozio_income_label AUTO
  sozio_job_status_label AUTO
  electronics_use_label AUTO
  games_use_label AUTO
  AUD F2
  TUD F2
  IDP_MaxCORR F3
  IDP_CORR F3
  IDP_ERR F3
  IDP_Missing F3
  IDP_TooFast F3
  IDP_PER F10.2
  IDP_ACC F10.2
  IDP_RT F10.2
  IDP_RT_SD F10.2
  SAW_MaxCORR F3
  SAW_CORR F3
  SAW_ERR F3
  SAW_Missing F3
  SAW_TooFast F3
  SAW_PER F10.2
  SAW_ACC F10.2
  SAW_RT F10.2
  SAW_RT_SD F10.2
  SWM_MaxCORR F3
  SWM_CORR F3
  SWM_ERR F3
  SWM_Missing F3
  SWM_TooFast F3
  SWM_PER F10.2
  SWM_ACC F10.2
  SWM_RT F10.2
  SWM_RT_SD F10.2
  SER_MaxCORR F3
  SER_CORR F3
  SER_ERR F3
  SER_Missing F3
  SER_TooFast F3
  SER_PER F10.2
  SER_ACC F10.2
  SER_RT F10.2
  SER_RT_SD F10.2
  COMP_ACC F10.2
  COMP_RT F10.2
  /MAP.
RESTORE.
CACHE.
EXECUTE.
DATASET NAME SAT_subjectLevel WINDOW=FRONT.

ADD VALUE LABELS group_id 0'HC_match_all' 10'HC_match_AUD_only' 11'HC_match_TUD_only' 12'HC_match_AUD_and_TUD' 20'AUD_only' 21'TUD_only' 22'AUD_and_TUD'.
ADD VALUE LABELS gender 0'male' 1'female'.
ADD VALUE LABELS SAT_keys_condition 0'normal' 1'switched'.

VARIABLE LABELS 
    IDP_PER 'IDP_PER (%)' IDP_RT 'IDP_RT (s)' IDP_RT_SD 'IDP_RT_SD (s)' 
    SAW_PER 'SAW_PER (%)' SAW_RT 'SAW_RT (s)' SAW_RT_SD 'SAW_RT_SD (s)' 
    SWM_PER 'SWM_PER (%)' SWM_RT 'SWM_RT (s)' SWM_RT_SD 'SWM_RT_SD (s)' 
    SAT_PER 'SAT_PER (%)' SAT_RT 'SAT_RT (s)' MeanPD 'Mean Planning Depth'.

VARIABLE LEVEL 
    SAT_Total_points 
    group_id 
    RAV_MaxCORR 
    RAV_CORR 
    RAV_ERR 
    RAV_Missing 
    RAV_TooFast 
    RAV_CORR_Instr 
    ftnd_sum 
    audit_score 
    age 
    nfc_score
    bis15_sum_planning
    bis15_sum_motor
    bis15_sum_attention
    bis15_sum_all
    qf_tab_3m_smokingDays 
    qf_tab_3m_meanDailyCigarettes 
    qf_alc_3m_drinkingDays 
    qf_alc_3m_bingingDays 
    IDP_MaxCORR 
    IDP_CORR 
    IDP_ERR 
    IDP_Missing 
    IDP_TooFast 
    SAW_MaxCORR 
    SAW_CORR 
    SAW_ERR 
    SAW_Missing 
    SAW_TooFast 
    SWM_MaxCORR 
    SWM_CORR 
    SWM_ERR 
    SWM_Missing 
    SWM_TooFast 
    SER_MaxCORR 
    SER_CORR 
    SER_ERR 
    SER_Missing 
    SER_TooFast (SCALE)
    /scid_tud_sum 
    scid_aud_sum 
    sozio_income 
    electronics_use 
    games_use (ORDINAL)
    /SAT_keys_condition 
    group_id gender
    sozio_income_label 
    electronics_use_label     
    games_use_label 
    qf_alc_3m_drinkingDays_sumOrWeekly 
    qf_alc_3m_bingingDays_sumOrWeekly 
    qf_tab_3m_smokingDays_sumOrWeekly 
    qf_tab_lifetime_smoker (NOMINAL).

addValueLabelsFromLabelCol 
    indexCol = sozio_income 
    labelCol = sozio_income_label 
    indexColStr = "sozio_income"
    datasetname = SAT_subjectLevel.

addValueLabelsFromLabelCol 
    indexCol = sozio_job_status 
    labelCol = sozio_job_status_label 
    indexColStr = "sozio_job_status"
    datasetname = SAT_subjectLevel.

addValueLabelsFromLabelCol 
    indexCol = sozio_graduat 
    labelCol = sozio_graduat_label 
    indexColStr = "sozio_graduat"
    datasetname = SAT_subjectLevel.

addValueLabelsFromLabelCol 
    indexCol = games_use 
    labelCol = games_use_label 
    indexColStr = "games_use"
    datasetname = SAT_subjectLevel.

addValueLabelsFromLabelCol 
    indexCol = electronics_use 
    labelCol = electronics_use_label 
    indexColStr = "electronics_use"
    datasetname = SAT_subjectLevel.

DELETE VARIABLES sozio_income_label sozio_job_status_label sozio_graduat_label  games_use_label  electronics_use_label WAIC_pointwise pWAIC_pointwise WAIC_marginal DIC.
COMPUTE group2_id = (CHAR.INDEX(UPCASE(group2),'SUD') > 0).
COMPUTE model_k_ln =LN(model_k).
EXECUTE.
ADD VALUE LABELS group2_id 0'matched_HC' 1'SUD'.

DATASET ACTIVATE SAT_subjectLevel.
COMPUTE BIC=(BIC_hinoise_mean + BIC_lonoise_mean) / 2.
EXECUTE.

* compute total consumption days for tobacco and alcohol over the last 3 months

DATASET ACTIVATE SAT_subjectLevel.
COMPUTE qf_tab_3m_smokingDays_sum = qf_tab_3m_smokingDays + (qf_tab_3m_smokingDays_sumOrWeekly-1) * 
    11.5 * qf_tab_3m_smokingDays.
EXECUTE.
COMPUTE qf_alc_3m_drinkingDays_sum = qf_alc_3m_drinkingDays + (qf_alc_3m_drinkingDays_sumOrWeekly-1) * 
    11.5 * qf_alc_3m_drinkingDays.
EXECUTE.
COMPUTE qf_alc_3m_bingingDays_sum = qf_alc_3m_bingingDays + (qf_alc_3m_bingingDays_sumOrWeekly-1) * 
    11.5 * qf_alc_3m_bingingDays.
EXECUTE.

* compute interaction term

COMPUTE AUDxTUD=AUD * TUD.
EXECUTE.

SAVE OUTFILE='SAT_subjectLevel.sav'
  /COMPRESSED.




*** load miniblock-level CSV data
    

PRESERVE.
SET DECIMAL DOT.

GET DATA  /TYPE=TXT
  /FILE="SAT_singleMiniblocks.csv"
  /ENCODING='UTF8'
  /DELIMITERS=";"
  /QUALIFIER="'"
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /DATATYPEMIN PERCENTAGE=95.0
  /VARIABLES=
  ID A5
  group AUTO
  block_num F3
  noise F1
  steps F1
  a0 F10.3
  SAT_RT F10.2
  MeanPD F10.3
  SAT_keys_condition F1
  SAT_Total_points F4
  SAT_PER F10.2
  group_id F2
  subject 4X
  group2 AUTO
  condition AUTO
  model_beta F10.3
  model_theta F10.3
  model_k F10.3
  WAIC F10.3
  WAIC_lowNoise F10.3
  WAIC_highNoise F10.3
  pWAIC F10.3
  DIC F10.3
  NLL_sampled F10.3
  nll_1staction_mean 17X
  nll_1staction_hinoise_mean 17X
  nll_1staction_lonoise_mean 18X
  BIC_mean 18X
  BIC_hinoise_mean F10.3
  BIC_lonoise_mean F10.3
  pseudo_Rsquare_1staction_mean 19X
  pseudo_Rsquare_1staction_hinoise_mean 19X
  pseudo_Rsquare_1staction_lonoise_mean 18X
  WAIC_marginal F10.3
  WAIC_pointwise F10.3
  pWAIC_pointwise F10.3
  RAV_MaxCORR F3
  RAV_CORR F3
  RAV_ERR F3
  RAV_Missing F3
  RAV_TooFast F3
  RAV_PER F10.2
  RAV_ACC F10.2
  RAV_RT F10.2
  RAV_RT_SD F10.2
  RAV_RT_Instr F10.2
  RAV_RT_Task F10.2
  RAV_CORR_Instr F3
  scid_tud_sum F2
  scid_aud_sum F2
  ftnd_sum F2
  audit_score F2
  age F3
  gender F2
  qf_tab_3m_smokingDays F3
  qf_tab_3m_smokingDays_sumOrWeekly F3
  qf_tab_3m_meanDailyCigarettes F3
  qf_tab_lifetime_smoker F3
  qf_alc_3m_drinkingDays F3
  qf_alc_3m_drinkingDays_sumOrWeekly F3
  qf_alc_3m_bingingDays F3
  qf_alc_3m_bingingDays_sumOrWeekly F3
  qf_alc_amount_perDay F10.2
  qf_alc_amount_perWorkday F10.2
  qf_alc_amount_perWeekendDay F10.2
  qf_alc_amount_lastDay F10.2
  group_HC F2
  group_AUD_only F2
  group_TUD_only F2
  group_AUD_and_TUD F2
  sozio_income F2
  sozio_graduat F2
  sozio_graduat_other F2
  electronics_use F2
  games_use F2
  working F2
  sozio_job_status F2
  nfc_score F2
  bis15_sum_planning F2
  bis15_sum_motor F2
  bis15_sum_attention F2
  bis15_sum_all F2
  gender_label AUTO
  sozio_graduat_label AUTO
  HEEQ F2
  sozio_income_label AUTO
  sozio_job_status_label AUTO
  electronics_use_label AUTO
  games_use_label AUTO
  AUD F2
  TUD F2
  IDP_MaxCORR F3
  IDP_CORR F3
  IDP_ERR F3
  IDP_Missing F3
  IDP_TooFast F3
  IDP_PER F10.2
  IDP_ACC F10.2
  IDP_RT F10.2
  IDP_RT_SD F10.2
  SAW_MaxCORR F3
  SAW_CORR F3
  SAW_ERR F3
  SAW_Missing F3
  SAW_TooFast F3
  SAW_PER F10.2
  SAW_ACC F10.2
  SAW_RT F10.2
  SAW_RT_SD F10.2
  SWM_MaxCORR F3
  SWM_CORR F3
  SWM_ERR F3
  SWM_Missing F3
  SWM_TooFast F3
  SWM_PER F10.2
  SWM_ACC F10.2
  SWM_RT F10.2
  SWM_RT_SD F10.2
  SER_MaxCORR F3
  SER_CORR F3
  SER_ERR F3
  SER_Missing F3
  SER_TooFast F3
  SER_PER F10.2
  SER_ACC F10.2
  SER_RT F10.2
  SER_RT_SD F10.2
  COMP_ACC F10.2
  COMP_RT F10.2
  /MAP.
RESTORE.
CACHE.
EXECUTE.
DATASET NAME SAT_singleMiniblocks WINDOW=FRONT.

ADD VALUE LABELS group_id 0'HC_match_all' 10'HC_match_AUD_only' 11'HC_match_TUD_only' 12'HC_match_AUD_and_TUD' 20'AUD_only' 21'TUD_only' 22'AUD_and_TUD'.
ADD VALUE LABELS gender 0'male' 1'female'.
ADD VALUE LABELS noise 0'low' 1'high'.
ADD VALUE LABELS SAT_keys_condition 0'normal' 1'switched'.

VARIABLE LABELS 
    IDP_PER 'IDP_PER (%)' IDP_RT 'IDP_RT (s)' IDP_RT_SD 'IDP_RT_SD (s)' 
    SAW_PER 'SAW_PER (%)' SAW_RT 'SAW_RT (s)' SAW_RT_SD 'SAW_RT_SD (s)' 
    SWM_PER 'SWM_PER (%)' SWM_RT 'SWM_RT (s)' SWM_RT_SD 'SWM_RT_SD (s)' 
    SAT_PER 'SAT_PER (%)' SAT_RT 'SAT_RT (s)' MeanPD 'Mean Planning Depth'.

VARIABLE LEVEL 
    SAT_Total_points 
    group_id 
    RAV_MaxCORR 
    RAV_CORR 
    RAV_ERR 
    RAV_Missing 
    RAV_TooFast 
    RAV_CORR_Instr 
    ftnd_sum 
    audit_score 
    age 
    nfc_score
    bis15_sum_planning
    bis15_sum_motor
    bis15_sum_attention
    bis15_sum_all
    qf_tab_3m_smokingDays 
    qf_tab_3m_meanDailyCigarettes 
    qf_alc_3m_drinkingDays 
    qf_alc_3m_bingingDays 
    IDP_MaxCORR 
    IDP_CORR 
    IDP_ERR 
    IDP_Missing 
    IDP_TooFast 
    SAW_MaxCORR 
    SAW_CORR 
    SAW_ERR 
    SAW_Missing 
    SAW_TooFast 
    SWM_MaxCORR 
    SWM_CORR 
    SWM_ERR 
    SWM_Missing 
    SWM_TooFast 
    SER_MaxCORR 
    SER_CORR 
    SER_ERR 
    SER_Missing 
    SER_TooFast (SCALE)
    /scid_tud_sum 
    scid_aud_sum 
    sozio_income 
    electronics_use 
    games_use (ORDINAL)
    /SAT_keys_condition 
    group_id gender
    sozio_income_label 
    electronics_use_label     
    games_use_label 
    qf_alc_3m_drinkingDays_sumOrWeekly 
    qf_alc_3m_bingingDays_sumOrWeekly 
    qf_tab_3m_smokingDays_sumOrWeekly 
    qf_tab_lifetime_smoker (NOMINAL).

addValueLabelsFromLabelCol 
    indexCol = sozio_income 
    labelCol = sozio_income_label 
    indexColStr = "sozio_income"
    datasetname = SAT_singleMiniblocks.

addValueLabelsFromLabelCol 
    indexCol = sozio_job_status 
    labelCol = sozio_job_status_label 
    indexColStr = "sozio_job_status"
    datasetname = SAT_singleMiniblocks.

addValueLabelsFromLabelCol 
    indexCol = sozio_graduat 
    labelCol = sozio_graduat_label 
    indexColStr = "sozio_graduat"
    datasetname = SAT_singleMiniblocks.

addValueLabelsFromLabelCol 
    indexCol = games_use 
    labelCol = games_use_label 
    indexColStr = "games_use"
    datasetname = SAT_singleMiniblocks.

addValueLabelsFromLabelCol 
    indexCol = electronics_use 
    labelCol = electronics_use_label 
    indexColStr = "electronics_use"
    datasetname = SAT_singleMiniblocks.

DELETE VARIABLES sozio_income_label sozio_job_status_label sozio_graduat_label  games_use_label  electronics_use_label WAIC_marginal DIC.

COMPUTE group2_id = (CHAR.INDEX(UPCASE(group2),'SUD') > 0).
COMPUTE model_k_ln=LN(model_k).
EXECUTE.
ADD VALUE LABELS group2_id 0'matched_HC' 1'SUD'.

DATASET ACTIVATE SAT_singleMiniblocks.
COMPUTE BIC=(BIC_hinoise_mean + BIC_lonoise_mean) / 2.
EXECUTE.

SAVE OUTFILE='SAT_singleMiniblocks.sav'
  /COMPRESSED.





*** OR LOAD SPSS DATA FILES.

GET  FILE='SAT_subjectLevel.sav'.
DATASET NAME SAT_subjectLevel WINDOW=FRONT.

GET  FILE='SAT_singleMiniblocks.sav'.
DATASET NAME SAT_singleMiniblocks WINDOW=FRONT.





*** Run Analysis seperate for each SUD group compared to resp. matched_HC

DATASET ACTIVATE SAT_subjectLevel.
FILTER OFF.
SORT CASES  BY condition.
SPLIT FILE SEPARATE BY condition.


*** group comparison gender.

DATASET ACTIVATE SAT_subjectLevel.
CROSSTABS
  /TABLES=group2_id BY gender
  /FORMAT=AVALUE TABLES
  /STATISTICS=CHISQ 
  /CELLS=COUNT
  /COUNT ROUND CELL.


*** group comparison electronics and computer use

DATASET ACTIVATE SAT_subjectLevel.
NPAR TESTS
  /M-W= electronics_use games_use BY group2_id(0 1)
  /MISSING ANALYSIS.



*** group comparison higher education entrance qualification.

DATASET ACTIVATE SAT_subjectLevel.
CROSSTABS
  /TABLES=group2_id BY HEEQ
  /FORMAT=AVALUE TABLES
  /STATISTICS=CHISQ 
  /CELLS=COUNT
  /COUNT ROUND CELL.


*** group comparison employment.

DATASET ACTIVATE SAT_subjectLevel.
CROSSTABS
  /TABLES=group2_id BY working
  /FORMAT=AVALUE TABLES
  /STATISTICS=CHISQ 
  /CELLS=COUNT
  /COUNT ROUND CELL.


*** group comparisons normality test.

DATASET ACTIVATE SAT_subjectLevel.
EXAMINE VARIABLES=age audit_score ftnd_sum qf_alc_3m_bingingDays_sum qf_tab_3m_meanDailyCigarettes SAT_PER SAT_RT MeanPD model_k model_k_ln model_beta model_theta SWM_PER SWM_RT SAW_PER SAW_RT IDP_PER IDP_RT RAV_PER 
  nfc_score electronics_use games_use BY group2_id
  /PLOT HISTOGRAM NPPLOT
  /COMPARE GROUPS
  /STATISTICS NONE
  /CINTERVAL 95
  /MISSING PAIRWISE
  /NOTOTAL.


*** group comparisons standard t-test.

DATASET ACTIVATE SAT_subjectLevel.
T-TEST GROUPS=group2_id(0 1)
  /MISSING=ANALYSIS
  /VARIABLES=age audit_score ftnd_sum MeanPD model_k model_k_ln model_beta model_theta qf_alc_3m_bingingDays_sum qf_tab_3m_meanDailyCigarettes SAT_PER SAW_PER SWM_PER IDP_PER RAV_PER SAT_RT SAW_RT SWM_RT IDP_RT nfc_score
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).



*** group comparisons bootstrap and mann whitney u.

* AUD_only

DATASET ACTIVATE SAT_subjectLevel.
USE ALL.
COMPUTE filter_condition = (condition = "AUD_only").
FILTER BY filter_condition.
SORT CASES  BY condition group2_id.
SPLIT FILE OFF.
EXECUTE.

TITLE 'AUD_only'.

DATASET ACTIVATE SAT_subjectLevel.
BOOTSTRAP
  /SAMPLING METHOD=SIMPLE
  /VARIABLES TARGET=age audit_score ftnd_sum qf_alc_3m_bingingDays_sum qf_tab_3m_meanDailyCigarettes  nfc_score games_use electronics_use
  MeanPD model_k model_k_ln model_beta model_theta SAT_PER SAW_PER SWM_PER IDP_PER RAV_PER SAT_RT SAW_RT SWM_RT IDP_RT
    INPUT=group2_id 
  /CRITERIA CILEVEL=95 CITYPE=BCA  NSAMPLES=100
  /MISSING USERMISSING=EXCLUDE.
T-TEST GROUPS=group2_id(0 1)
  /MISSING=ANALYSIS
  /VARIABLES=age audit_score ftnd_sum qf_alc_3m_bingingDays_sum qf_tab_3m_meanDailyCigarettes nfc_score games_use electronics_use 
  MeanPD model_k model_k_ln model_beta model_theta SAT_PER SAW_PER SWM_PER IDP_PER RAV_PER IDP_PER SAT_RT SAW_RT SWM_RT IDP_RT
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).

NPAR TESTS
  /M-W= age audit_score ftnd_sum qf_alc_3m_bingingDays_sum qf_tab_3m_meanDailyCigarettes nfc_score games_use electronics_use 
  MeanPD model_k model_k_ln model_beta model_theta SAT_PER SAW_PER SWM_PER IDP_PER RAV_PER SAT_RT SAW_RT SWM_RT IDP_RT BY group2_id(0 1)
  /MISSING ANALYSIS.

* TUD_only

DATASET ACTIVATE SAT_subjectLevel.
USE ALL.
COMPUTE filter_condition = (condition = "TUD_only").
FILTER BY filter_condition.
SORT CASES  BY condition group2_id.
SPLIT FILE OFF.
EXECUTE.

TITLE 'TUD_only'.

DATASET ACTIVATE SAT_subjectLevel.
BOOTSTRAP
  /SAMPLING METHOD=SIMPLE
  /VARIABLES TARGET=age audit_score ftnd_sum qf_alc_3m_bingingDays_sum qf_tab_3m_meanDailyCigarettes  nfc_score games_use electronics_use
  MeanPD model_k model_k_ln model_beta model_theta SAT_PER SAW_PER SWM_PER IDP_PER RAV_PER SAT_RT SAW_RT SWM_RT IDP_RT
    INPUT=group2_id 
  /CRITERIA CILEVEL=95 CITYPE=BCA  NSAMPLES=100
  /MISSING USERMISSING=EXCLUDE.
T-TEST GROUPS=group2_id(0 1)
  /MISSING=ANALYSIS
  /VARIABLES=age audit_score ftnd_sum qf_alc_3m_bingingDays_sum qf_tab_3m_meanDailyCigarettes  nfc_score games_use electronics_use
  MeanPD model_k model_k_ln model_beta model_theta SAT_PER SAW_PER SWM_PER IDP_PER RAV_PER SAT_RT SAW_RT SWM_RT IDP_RT
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).

NPAR TESTS
  /M-W= age audit_score ftnd_sum qf_alc_3m_bingingDays_sum qf_tab_3m_meanDailyCigarettes nfc_score games_use electronics_use 
  MeanPD model_k model_k_ln model_beta model_theta SAT_PER SAW_PER SWM_PER IDP_PER RAV_PER SAT_RT SAW_RT SWM_RT IDP_RT BY group2_id(0 1)
  /MISSING ANALYSIS.

* AUD_and_TUD

DATASET ACTIVATE SAT_subjectLevel.
USE ALL.
COMPUTE filter_condition = (condition = "AUD_and_TUD").
FILTER BY filter_condition.
SORT CASES  BY condition group2_id.
SPLIT FILE OFF.
EXECUTE.

TITLE 'AUD_and_TUD'.

DATASET ACTIVATE SAT_subjectLevel.
BOOTSTRAP
  /SAMPLING METHOD=SIMPLE
  /VARIABLES TARGET=age audit_score ftnd_sum qf_alc_3m_bingingDays_sum qf_tab_3m_meanDailyCigarettes  nfc_score games_use electronics_use
  MeanPD model_k model_k_ln model_beta model_theta SAT_PER SAW_PER SWM_PER IDP_PER RAV_PER SAT_RT SAW_RT SWM_RT IDP_RT
    INPUT=group2_id 
  /CRITERIA CILEVEL=95 CITYPE=BCA  NSAMPLES=100
  /MISSING USERMISSING=EXCLUDE.
T-TEST GROUPS=group2_id(0 1)
  /MISSING=ANALYSIS
  /VARIABLES=age audit_score ftnd_sum qf_alc_3m_bingingDays_sum qf_tab_3m_meanDailyCigarettes  nfc_score games_use electronics_use
  MeanPD model_k model_k_ln model_beta model_theta SAT_PER SAW_PER SWM_PER IDP_PER RAV_PER SAT_RT SAW_RT SWM_RT IDP_RT
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).

NPAR TESTS
  /M-W= age audit_score ftnd_sum qf_alc_3m_bingingDays_sum qf_tab_3m_meanDailyCigarettes nfc_score games_use electronics_use 
  MeanPD model_k model_k_ln model_beta model_theta SAT_PER SAW_PER SWM_PER IDP_PER RAV_PER SAT_RT SAW_RT SWM_RT IDP_RT BY group2_id(0 1)
  /MISSING ANALYSIS.


*** CORRELATIONS.
DATASET ACTIVATE SAT_subjectLevel.
COMPUTE filter_condition = ((group_id >= 20) | (group_id = 0)).
FILTER BY filter_condition.
SPLIT FILE OFF.
CORRELATIONS
  /VARIABLES=age working HEEQ SAT_PER SAT_RT qf_alc_3m_bingingDays_sum qf_tab_3m_meanDailyCigarettes MeanPD model_k model_k_ln model_beta model_theta SWM_PER SWM_RT SAW_PER SAW_RT IDP_PER IDP_RT RAV_PER 
  scid_aud_sum scid_tud_sum audit_score ftnd_sum nfc_score games_use electronics_use
  /PRINT=TWOTAIL NOSIG FULL
  /MISSING=PAIRWISE.
NONPAR CORR
  /VARIABLES=age working HEEQ SAT_PER SAT_RT qf_alc_3m_bingingDays_sum qf_tab_3m_meanDailyCigarettes MeanPD model_k model_k_ln model_beta model_theta SWM_PER SWM_RT SAW_PER SAW_RT IDP_PER IDP_RT RAV_PER 
  scid_aud_sum scid_tud_sum audit_score ftnd_sum nfc_score games_use electronics_use
  /PRINT=SPEARMAN TWOTAIL NOSIG FULL
  /MISSING=PAIRWISE.



*** LME model of MeanPD

* HC pooled (without group and group*noise interaction term)

DATASET ACTIVATE SAT_singleMiniblocks.
USE ALL.
COMPUTE filter_condition = (condition = "pooled").
FILTER BY filter_condition.
SORT CASES  BY condition group2_id.
SPLIT FILE OFF.
EXECUTE.

TITLE 'HC pooled'.

DATASET ACTIVATE SAT_singleMiniblocks.
MIXED MeanPD WITH noise
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, 
    ABSOLUTE)
  /FIXED= noise | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT noise  | SUBJECT(ID) COVTYPE(VC).
*  /SAVE = PRED(LME_PRED_hc) RESID(LME_RESID_hc).
*FORMATS LME_PRED_hc(F10.2) LME_RESID_hc(F10.2). 


* AUD_only

DATASET ACTIVATE SAT_singleMiniblocks.
USE ALL.
COMPUTE filter_condition = (condition = "AUD_only").
FILTER BY filter_condition.
SORT CASES  BY condition group2_id.
SPLIT FILE OFF.
EXECUTE.

TITLE 'AUD_only'.

DATASET ACTIVATE SAT_singleMiniblocks.
MIXED MeanPD WITH group2_id noise
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, 
    ABSOLUTE)
  /FIXED=group2_id noise  group2_id*noise | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT noise  | SUBJECT(ID) COVTYPE(VC).
*  /SAVE = PRED(LME_PRED_aud) RESID(LME_RESID_aud).
*FORMATS LME_PRED_aud(F10.2) LME_RESID_aud(F10.2). 


* TUD_only

DATASET ACTIVATE SAT_singleMiniblocks.
USE ALL.
COMPUTE filter_condition = (condition = "TUD_only").
FILTER BY filter_condition.
SORT CASES  BY condition group2_id.
SPLIT FILE OFF.
EXECUTE.

TITLE 'TUD_only'.

DATASET ACTIVATE SAT_singleMiniblocks.
MIXED MeanPD WITH group2_id noise
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, 
    ABSOLUTE)
  /FIXED=group2_id noise  group2_id*noise | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT noise  | SUBJECT(ID) COVTYPE(VC).
*  /SAVE = PRED(LME_PRED_tud) RESID(LME_RESID_tud).
*FORMATS LME_PRED_tud(F10.2) LME_RESID_tud(F10.2). 


* AUD_and_TUD

DATASET ACTIVATE SAT_singleMiniblocks.
USE ALL.
COMPUTE filter_condition = (condition = "AUD_and_TUD").
FILTER BY filter_condition.
SORT CASES  BY condition group2_id.
SPLIT FILE OFF.
EXECUTE.

TITLE 'AUD_and_TUD'.

DATASET ACTIVATE SAT_singleMiniblocks.
MIXED MeanPD WITH group2_id noise
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, 
    ABSOLUTE)
  /FIXED=group2_id noise  group2_id*noise | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT noise  | SUBJECT(ID) COVTYPE(VC).
*  /SAVE = PRED(LME_PRED_atud) RESID(LME_RESID_atud).
*FORMATS LME_PRED_atud(F10.2) LME_RESID_atud(F10.2). 






*** LME model of SAT_RT

* HC pooled (without group and group*noise interaction term)

DATASET ACTIVATE SAT_singleMiniblocks.
USE ALL.
COMPUTE filter_condition = (condition = "pooled").
FILTER BY filter_condition.
SORT CASES  BY condition group2_id.
SPLIT FILE OFF.
EXECUTE.

TITLE 'HC pooled'.

DATASET ACTIVATE SAT_singleMiniblocks.
MIXED SAT_RT WITH noise
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, 
    ABSOLUTE)
  /FIXED= noise | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT noise  | SUBJECT(ID) COVTYPE(VC).
*  /SAVE = PRED(LME_PRED_hc) RESID(LME_RESID_hc).
*FORMATS LME_PRED_hc(F10.2) LME_RESID_hc(F10.2). 


* AUD_only

DATASET ACTIVATE SAT_singleMiniblocks.
USE ALL.
COMPUTE filter_condition = (condition = "AUD_only").
FILTER BY filter_condition.
SORT CASES  BY condition group2_id.
SPLIT FILE OFF.
EXECUTE.

TITLE 'AUD_only'.

DATASET ACTIVATE SAT_singleMiniblocks.
MIXED SAT_RT WITH group2_id noise
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, 
    ABSOLUTE)
  /FIXED=group2_id noise  group2_id*noise | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT noise  | SUBJECT(ID) COVTYPE(VC).
*  /SAVE = PRED(LME_PRED_aud) RESID(LME_RESID_aud).
*FORMATS LME_PRED_aud(F10.2) LME_RESID_aud(F10.2). 


* TUD_only

DATASET ACTIVATE SAT_singleMiniblocks.
USE ALL.
COMPUTE filter_condition = (condition = "TUD_only").
FILTER BY filter_condition.
SORT CASES  BY condition group2_id.
SPLIT FILE OFF.
EXECUTE.

TITLE 'TUD_only'.

DATASET ACTIVATE SAT_singleMiniblocks.
MIXED SAT_RT WITH group2_id noise
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, 
    ABSOLUTE)
  /FIXED=group2_id noise  group2_id*noise | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT noise  | SUBJECT(ID) COVTYPE(VC).
*  /SAVE = PRED(LME_PRED_tud) RESID(LME_RESID_tud).
*FORMATS LME_PRED_tud(F10.2) LME_RESID_tud(F10.2). 


* AUD_and_TUD

DATASET ACTIVATE SAT_singleMiniblocks.
USE ALL.
COMPUTE filter_condition = (condition = "AUD_and_TUD").
FILTER BY filter_condition.
SORT CASES  BY condition group2_id.
SPLIT FILE OFF.
EXECUTE.

TITLE 'AUD_and_TUD'.

DATASET ACTIVATE SAT_singleMiniblocks.
MIXED SAT_RT WITH group2_id noise
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, 
    ABSOLUTE)
  /FIXED=group2_id noise  group2_id*noise | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT noise  | SUBJECT(ID) COVTYPE(VC).
*  /SAVE = PRED(LME_PRED_atud) RESID(LME_RESID_atud).
*FORMATS LME_PRED_atud(F10.2) LME_RESID_atud(F10.2). 







*** LME model of SAT_gain

* HC pooled (without group and group*noise interaction term)

DATASET ACTIVATE SAT_singleMiniblocks.
USE ALL.
COMPUTE filter_condition = (condition = "pooled").
FILTER BY filter_condition.
SORT CASES  BY condition group2_id.
SPLIT FILE OFF.
EXECUTE.

TITLE 'HC pooled'.

DATASET ACTIVATE SAT_singleMiniblocks.
MIXED SAT_gain WITH noise
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, 
    ABSOLUTE)
  /FIXED= noise | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT noise  | SUBJECT(ID) COVTYPE(VC).
*  /SAVE = PRED(LME_PRED_hc) RESID(LME_RESID_hc).
*FORMATS LME_PRED_hc(F10.2) LME_RESID_hc(F10.2). 


* AUD_only

DATASET ACTIVATE SAT_singleMiniblocks.
USE ALL.
COMPUTE filter_condition = (condition = "AUD_only").
FILTER BY filter_condition.
SORT CASES  BY condition group2_id.
SPLIT FILE OFF.
EXECUTE.

TITLE 'AUD_only'.

DATASET ACTIVATE SAT_singleMiniblocks.
MIXED SAT_gain WITH group2_id noise
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, 
    ABSOLUTE)
  /FIXED=group2_id noise  group2_id*noise | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT noise  | SUBJECT(ID) COVTYPE(VC).
*  /SAVE = PRED(LME_PRED_aud) RESID(LME_RESID_aud).
*FORMATS LME_PRED_aud(F10.2) LME_RESID_aud(F10.2). 


* TUD_only

DATASET ACTIVATE SAT_singleMiniblocks.
USE ALL.
COMPUTE filter_condition = (condition = "TUD_only").
FILTER BY filter_condition.
SORT CASES  BY condition group2_id.
SPLIT FILE OFF.
EXECUTE.

TITLE 'TUD_only'.

DATASET ACTIVATE SAT_singleMiniblocks.
MIXED SAT_gain WITH group2_id noise
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, 
    ABSOLUTE)
  /FIXED=group2_id noise  group2_id*noise | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT noise  | SUBJECT(ID) COVTYPE(VC).
*  /SAVE = PRED(LME_PRED_tud) RESID(LME_RESID_tud).
*FORMATS LME_PRED_tud(F10.2) LME_RESID_tud(F10.2). 


* AUD_and_TUD

DATASET ACTIVATE SAT_singleMiniblocks.
USE ALL.
COMPUTE filter_condition = (condition = "AUD_and_TUD").
FILTER BY filter_condition.
SORT CASES  BY condition group2_id.
SPLIT FILE OFF.
EXECUTE.

TITLE 'AUD_and_TUD'.

DATASET ACTIVATE SAT_singleMiniblocks.
MIXED SAT_gain WITH group2_id noise
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, 
    ABSOLUTE)
  /FIXED=group2_id noise  group2_id*noise | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT noise  | SUBJECT(ID) COVTYPE(VC).
*  /SAVE = PRED(LME_PRED_atud) RESID(LME_RESID_atud).
*FORMATS LME_PRED_atud(F10.2) LME_RESID_atud(F10.2). 







*** linear regression MeanPD.

DATASET ACTIVATE SAT_subjectLevel.
COMPUTE filter_condition = (condition ~= "pooled").
FILTER BY filter_condition.
SORT CASES  BY condition.
SPLIT FILE SEPARATE BY condition.

DATASET ACTIVATE SAT_subjectLevel.
REGRESSION
  /DESCRIPTIVES MEAN STDDEV CORR SIG N
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS CI(95) R ANOVA COLLIN TOL CHANGE ZPP
  /CRITERIA=PIN(.05) POUT(.10)
  /NOORIGIN 
  /DEPENDENT MeanPD
  /METHOD=BACKWARD group2_id SAT_RT  IDP_PER SWM_PER RAV_PER.
*  /SAVE = PRED(LMQ_PRED) RESID(LMQ_RESID).
*FORMATS LMQ_PRED(F10.2) LMQ_RESID(F10.2).




*** model quality linear regression.

DATASET ACTIVATE SAT_subjectLevel.
COMPUTE filter_condition = ((group_id >= 20) | (group_id = 0)).
FILTER BY filter_condition.
SPLIT FILE OFF.


DATASET ACTIVATE SAT_subjectLevel.
REGRESSION
  /DESCRIPTIVES MEAN STDDEV CORR SIG N
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS CI(95) R ANOVA COLLIN TOL CHANGE ZPP
  /CRITERIA=PIN(.05) POUT(.10)
  /NOORIGIN 
  /DEPENDENT SAT_PER
  /METHOD=ENTER model_k_ln model_beta model_theta MeanPD.
*    /SAVE = PRED(LMQ_PRED) RESID(LMQ_RESID).
*FORMATS LMQ_PRED(F10.2) LMQ_RESID(F10.2).

DATASET ACTIVATE SAT_subjectLevel.
REGRESSION
  /DESCRIPTIVES MEAN STDDEV CORR SIG N
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS CI(95) R ANOVA COLLIN TOL CHANGE ZPP
  /CRITERIA=PIN(.05) POUT(.10)
  /NOORIGIN 
  /DEPENDENT SAT_RT
  /METHOD=ENTER model_k_ln model_beta model_theta MeanPD.
*    /SAVE = PRED(LMQ_PRED) RESID(LMQ_RESID).
*FORMATS LMQ_PRED(F10.2) LMQ_RESID(F10.2).




*** Factorial analyses of outcomes/parameters and AUD and TUD as factors with interaction term
   
DATASET ACTIVATE SAT_subjectLevel.
COMPUTE filter_condition = ((group_id >= 20) | (group_id = 0)).
FILTER BY filter_condition.
SPLIT FILE OFF.


BOOTSTRAP
  /SAMPLING METHOD=SIMPLE
  /VARIABLES TARGET=MeanPD INPUT=  AUD TUD AUDxTUD  
  /CRITERIA CILEVEL=95 CITYPE=BCA  NSAMPLES=1000
  /MISSING USERMISSING=EXCLUDE.
REGRESSION
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS R ANOVA
  /CRITERIA=PIN(.05) POUT(.10) TOLERANCE(.0001)
  /NOORIGIN 
  /DEPENDENT MeanPD
  /METHOD=ENTER AUD TUD AUDxTUD.

BOOTSTRAP
  /SAMPLING METHOD=SIMPLE
  /VARIABLES TARGET=SAT_RT INPUT=  AUD TUD AUDxTUD  
  /CRITERIA CILEVEL=95 CITYPE=BCA  NSAMPLES=1000
  /MISSING USERMISSING=EXCLUDE.
REGRESSION
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS R ANOVA
  /CRITERIA=PIN(.05) POUT(.10) TOLERANCE(.0001)
  /NOORIGIN 
  /DEPENDENT SAT_RT
  /METHOD=ENTER AUD TUD AUDxTUD.


BOOTSTRAP
  /SAMPLING METHOD=SIMPLE
  /VARIABLES TARGET=SAT_PER INPUT=  AUD TUD AUDxTUD  
  /CRITERIA CILEVEL=95 CITYPE=BCA  NSAMPLES=1000
  /MISSING USERMISSING=EXCLUDE.
REGRESSION
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS R ANOVA
  /CRITERIA=PIN(.05) POUT(.10) TOLERANCE(.0001)
  /NOORIGIN 
  /DEPENDENT SAT_PER
  /METHOD=ENTER AUD TUD AUDxTUD.


BOOTSTRAP
  /SAMPLING METHOD=SIMPLE
  /VARIABLES TARGET=model_beta INPUT=  AUD TUD AUDxTUD  
  /CRITERIA CILEVEL=95 CITYPE=BCA  NSAMPLES=1000
  /MISSING USERMISSING=EXCLUDE.
REGRESSION
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS R ANOVA
  /CRITERIA=PIN(.05) POUT(.10) TOLERANCE(.0001)
  /NOORIGIN 
  /DEPENDENT model_beta
  /METHOD=ENTER AUD TUD AUDxTUD.


BOOTSTRAP
  /SAMPLING METHOD=SIMPLE
  /VARIABLES TARGET=model_k INPUT=  AUD TUD AUDxTUD  
  /CRITERIA CILEVEL=95 CITYPE=BCA  NSAMPLES=1000
  /MISSING USERMISSING=EXCLUDE.
REGRESSION
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS R ANOVA
  /CRITERIA=PIN(.05) POUT(.10) TOLERANCE(.0001)
  /NOORIGIN 
  /DEPENDENT model_k
  /METHOD=ENTER AUD TUD AUDxTUD.





* REMOVE all filters again

DATASET ACTIVATE SAT_subjectLevel.
USE ALL.
FILTER OFF.
SPLIT FILE OFF.
EXECUTE.

DATASET ACTIVATE SAT_singleMiniblocks.
USE ALL.
FILTER OFF.
SPLIT FILE OFF.
EXECUTE.

