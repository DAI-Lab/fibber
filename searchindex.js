Search.setIndex({docnames:["api/fibber","api/fibber.dataset","api/fibber.dataset.dataset_utils","api/fibber.dataset.process_ag","api/fibber.dataset.process_imdb","api/fibber.dataset.process_mnli","api/fibber.dataset.process_mr","api/fibber.dataset.process_snli","api/fibber.dataset.process_yelp","api/fibber.download_utils","api/fibber.downloadable_resources","api/fibber.log","api/fibber.measurement","api/fibber.measurement.bert_clf_prediction","api/fibber.measurement.editing_distance","api/fibber.measurement.glove_semantic_similarity","api/fibber.measurement.gpt2_grammar_quality","api/fibber.measurement.measurement_base","api/fibber.measurement.measurement_utils","api/fibber.measurement.use_semantic_similarity","api/fibber.pipeline","api/fibber.pipeline.benchmark","api/fibber.pipeline.download_datasets","api/fibber.pipeline.make_overview","api/fibber.resource_utils","api/fibber.strategy","api/fibber.strategy.identical_strategy","api/fibber.strategy.random_strategy","api/fibber.strategy.strategy_base","api/fibber.strategy.textfooler_strategy","authors","contributing","history","index","readme"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.viewcode":1,nbsphinx:3,sphinx:56},filenames:["api/fibber.rst","api/fibber.dataset.rst","api/fibber.dataset.dataset_utils.rst","api/fibber.dataset.process_ag.rst","api/fibber.dataset.process_imdb.rst","api/fibber.dataset.process_mnli.rst","api/fibber.dataset.process_mr.rst","api/fibber.dataset.process_snli.rst","api/fibber.dataset.process_yelp.rst","api/fibber.download_utils.rst","api/fibber.downloadable_resources.rst","api/fibber.log.rst","api/fibber.measurement.rst","api/fibber.measurement.bert_clf_prediction.rst","api/fibber.measurement.editing_distance.rst","api/fibber.measurement.glove_semantic_similarity.rst","api/fibber.measurement.gpt2_grammar_quality.rst","api/fibber.measurement.measurement_base.rst","api/fibber.measurement.measurement_utils.rst","api/fibber.measurement.use_semantic_similarity.rst","api/fibber.pipeline.rst","api/fibber.pipeline.benchmark.rst","api/fibber.pipeline.download_datasets.rst","api/fibber.pipeline.make_overview.rst","api/fibber.resource_utils.rst","api/fibber.strategy.rst","api/fibber.strategy.identical_strategy.rst","api/fibber.strategy.random_strategy.rst","api/fibber.strategy.strategy_base.rst","api/fibber.strategy.textfooler_strategy.rst","authors.rst","contributing.rst","history.rst","index.rst","readme.rst"],objects:{"":{fibber:[0,0,0,"-"]},"fibber.dataset":{dataset_utils:[2,0,0,"-"],process_ag:[3,0,0,"-"],process_imdb:[4,0,0,"-"],process_mnli:[5,0,0,"-"],process_mr:[6,0,0,"-"],process_snli:[7,0,0,"-"],process_yelp:[8,0,0,"-"]},"fibber.dataset.dataset_utils":{DatasetForBert:[2,1,1,""],get_dataset:[2,2,1,""],subsample_dataset:[2,2,1,""],text_md5:[2,2,1,""],verify_dataset:[2,2,1,""]},"fibber.dataset.process_ag":{download_and_process_ag:[3,2,1,""],process_data:[3,2,1,""]},"fibber.dataset.process_imdb":{download_and_process_imdb:[4,2,1,""],process_data:[4,2,1,""]},"fibber.dataset.process_mnli":{download_and_process_mnli:[5,2,1,""],process_data:[5,2,1,""]},"fibber.dataset.process_mr":{download_and_process_mr:[6,2,1,""]},"fibber.dataset.process_snli":{download_and_process_snli:[7,2,1,""],process_data:[7,2,1,""]},"fibber.dataset.process_yelp":{download_and_process_yelp:[8,2,1,""],process_data:[8,2,1,""]},"fibber.download_utils":{check_file_md5:[9,2,1,""],download_file:[9,2,1,""],get_root_dir:[9,2,1,""]},"fibber.log":{add_filehandler:[11,2,1,""],setup_custom_logger:[11,2,1,""]},"fibber.measurement":{bert_clf_prediction:[13,0,0,"-"],editing_distance:[14,0,0,"-"],glove_semantic_similarity:[15,0,0,"-"],gpt2_grammar_quality:[16,0,0,"-"],measurement_base:[17,0,0,"-"],measurement_utils:[18,0,0,"-"],use_semantic_similarity:[19,0,0,"-"]},"fibber.measurement.bert_clf_prediction":{BertClfPrediction:[13,1,1,""],get_optimizer:[13,2,1,""],load_or_train_bert_clf:[13,2,1,""],run_evaluate:[13,2,1,""]},"fibber.measurement.bert_clf_prediction.BertClfPrediction":{predict:[13,3,1,""],predict_raw:[13,3,1,""]},"fibber.measurement.editing_distance":{EditingDistance:[14,1,1,""]},"fibber.measurement.glove_semantic_similarity":{GloVeSemanticSimilarity:[15,1,1,""],compute_emb:[15,2,1,""],compute_emb_sim:[15,2,1,""]},"fibber.measurement.gpt2_grammar_quality":{GPT2GrammarQuality:[16,1,1,""],make_batch:[16,2,1,""],make_input_output_pair:[16,2,1,""]},"fibber.measurement.measurement_base":{MeasurementBase:[17,1,1,""]},"fibber.measurement.measurement_utils":{MeasurementBundle:[18,1,1,""],aggregate_measurements:[18,2,1,""],majority_aggregation_fn:[18,2,1,""],mean_aggregation_fn:[18,2,1,""],measure_quality:[18,2,1,""],std_aggregation_fn:[18,2,1,""]},"fibber.measurement.measurement_utils.MeasurementBundle":{get_classifier_for_attack:[18,3,1,""]},"fibber.measurement.use_semantic_similarity":{USESemanticSimilarity:[19,1,1,""],config_tf_gpu:[19,2,1,""]},"fibber.pipeline":{benchmark:[21,0,0,"-"],download_datasets:[22,0,0,"-"],make_overview:[23,0,0,"-"]},"fibber.pipeline.benchmark":{benchmark:[21,2,1,""],get_output_filename:[21,2,1,""],get_strategy:[21,2,1,""],paraphrase_pred_accuracy_agg_fn:[21,2,1,""]},"fibber.pipeline.make_overview":{make_overview:[23,2,1,""]},"fibber.resource_utils":{get_glove_emb:[24,2,1,""],get_stopwords:[24,2,1,""],load_detailed_result:[24,2,1,""],load_glove_model:[24,2,1,""],update_detailed_result:[24,2,1,""],update_overview_result:[24,2,1,""]},"fibber.strategy":{identical_strategy:[26,0,0,"-"],random_strategy:[27,0,0,"-"],strategy_base:[28,0,0,"-"],textfooler_strategy:[29,0,0,"-"]},"fibber.strategy.identical_strategy":{IdenticalStrategy:[26,1,1,""]},"fibber.strategy.identical_strategy.IdenticalStrategy":{paraphrase_example:[26,3,1,""]},"fibber.strategy.random_strategy":{RandomStrategy:[27,1,1,""]},"fibber.strategy.random_strategy.RandomStrategy":{paraphrase_example:[27,3,1,""]},"fibber.strategy.strategy_base":{StrategyBase:[28,1,1,""]},"fibber.strategy.strategy_base.StrategyBase":{add_parser_args:[28,3,1,""],fit:[28,3,1,""],paraphrase:[28,3,1,""],paraphrase_example:[28,3,1,""]},"fibber.strategy.textfooler_strategy":{CLFModel:[29,1,1,""],TextFoolerStrategy:[29,1,1,""],compute_clf:[29,2,1,""],tostring:[29,2,1,""]},"fibber.strategy.textfooler_strategy.CLFModel":{set_context:[29,3,1,""],tokenize:[29,3,1,""]},"fibber.strategy.textfooler_strategy.TextFoolerStrategy":{paraphrase_example:[29,3,1,""]},fibber:{dataset:[1,0,0,"-"],download_utils:[9,0,0,"-"],downloadable_resources:[10,0,0,"-"],log:[11,0,0,"-"],measurement:[12,0,0,"-"],pipeline:[20,0,0,"-"],resource_utils:[24,0,0,"-"],strategy:[25,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","function","Python function"],"3":["py","method","Python method"]},objtypes:{"0":"py:module","1":"py:class","2":"py:function","3":"py:method"},terms:{"001":13,"0882519":[33,34],"100":[13,33,34],"10k":[33,34],"120k":[33,34],"160k":[33,34],"20000":[13,33,34],"2008":[33,34],"2014":24,"25k":[33,34],"300":24,"3342925":[33,34],"38k":[33,34],"4195759":[33,34],"433k":[33,34],"4389165":[33,34],"500":13,"5000":13,"570k":[33,34],"6441136":[33,34],"760386":[33,34],"8038205":[33,34],"80939928":[33,34],"81349877":[33,34],"84152539":[33,34],"8548079":[33,34],"8590095":[33,34],"8768223":[33,34],"91529946":[33,34],"break":[33,34],"case":[31,33,34],"class":[2,13,14,15,16,17,18,19,26,27,28,29,33,34],"default":24,"function":[26,27,28,29,31],"new":[24,31,33,34],"return":[2,24,26,27,28,29],"short":[33,34],"true":[14,18,33,34],"try":31,"while":31,AWS:[33,34],And:[31,33,34],For:[31,33,34],The:[28,31,33,34],Then:[33,34],These:[33,34],USE:[33,34],With:[33,34],__init__:[28,31],_strategy_config:28,a_modul:31,about:[31,33,34],action:31,activ:[33,34],actual:[31,33,34],adamw:13,add:[24,28,31],add_filehandl:11,add_parser_arg:28,after:[31,33,34],afterward:[33,34],aggreg:24,aggregate_measur:18,aggregated_result:24,algorithm:[33,34],all:[28,31,33,34],also:[31,33,34],although:[33,34],alwai:31,ani:[31,33,34],anyth:31,api:31,appl:[33,34],appreci:31,appropri:31,arg:28,argument:[33,34],arrai:24,articl:31,assert:2,assign:[33,34],associ:31,assum:31,avail:31,avoid:[33,34],back:29,base:[2,13,14,15,16,17,18,19,26,27,28,29,31],baselin:[26,27],batch_encod:29,batch_siz:2,becaus:[33,34],been:[31,33,34],befor:31,behavior:31,being:31,benchmark:[0,20],bert:[2,13,33,34],bert_clf:[33,34],bert_clf_b:13,bert_clf_lr:13,bert_clf_optim:13,bert_clf_period_sav:13,bert_clf_period_summari:13,bert_clf_period_v:13,bert_clf_predict:[0,12],bert_clf_step:[13,33,34],bert_clf_val_step:13,bert_clf_weight_decai:13,bert_gpu:[33,34],bert_gpu_id:13,bertclfpredict:13,best:31,between:[14,33,34],bit:31,blog:31,blown:31,bool:[33,34],boston:[33,34],both:31,branch:[31,33,34],briefli:31,bugfix:31,build:31,built:[33,34],bump:31,bumpvers:31,busi:[33,34],call:[29,31],can:[31,33,34],categori:2,cell:[33,34],cfg:31,championship:[33,34],chang:[31,33,34],check:[31,33,34],check_file_md5:9,checkout:[31,33,34],choos:[33,34],classif:[33,34],classifi:[33,34],classmethod:28,clf_measur:29,clf_model:29,clfmodel:29,clone:[31,33,34],code:[31,33,34],com:[31,33,34],combin:31,command:[28,31,33,34],commandlin:28,comment:31,commit:31,compli:31,compute_clf:29,compute_emb:15,compute_emb_sim:15,conda:[33,34],config:28,config_tf_gpu:19,configur:[28,33,34],consequ:31,consol:[33,34],contain:[2,26,27,28,29,31,33,34],context:29,contradict:[33,34],conveni:31,copi:31,correspond:31,could:31,cover:31,coverag:31,creat:[24,28,31,33,34],credit:31,csv:[33,34],current:31,customize_metr:18,customized_measur:18,dai:[33,34],data:[2,26,27,28,29,33,34],data_record:[26,27,28,29],databas:31,datafram:24,dataloader_it:13,dataset:[0,24,26,27,28,29],dataset_nam:[2,13,18,21,24,33,34],dataset_util:[0,1],datasetforbert:2,datasset:28,date:[33,34],decai:13,depend:31,deriv:28,descript:31,detail:[24,28,31,33,34],dev:31,develop:[31,33,34],devic:13,devn:31,dict:24,dictionari:[2,24,26,27,28,29],differ:[33,34],dim:24,dimens:24,dimension:24,distanc:14,doc:31,docstr:[18,31],document:[33,34],doe:[24,31],done:31,download:24,download_and_process_ag:3,download_and_process_imdb:4,download_and_process_mnli:5,download_and_process_mr:6,download_and_process_snli:7,download_and_process_yelp:8,download_dataset:[0,20,33,34],download_fil:9,download_util:0,downloadable_resourc:0,driven:31,dure:[28,33,34],each:[2,28,29,31,33,34],easier:31,easili:[33,34],edit:14,editing_dist:[0,12],editing_distance_ignore_punctu:14,editingdist:14,edu:[24,30],effect:[33,34],either:31,emb_tabl:[15,24],embed:24,empti:24,encod:[29,33,34],enhanc:31,entail:[33,34],entri:31,environ:[33,34],error:2,especi:[33,34],eval_step:13,evalu:[33,34],even:31,everi:[31,33,34],exampl:[2,28,33,34],exclud:2,execut:[31,33,34],exist:24,exp:[33,34],experiment_nam:18,explain:31,fail:31,fall:29,fals:[2,9,18],far:31,featur:[33,34],feel:31,fibber:31,fibber_env:[33,34],field:[2,26,27,28,29,33,34],field_nam:[26,27,28,29],file:[24,31,33,34],filenam:[9,11,24,28],first:[30,33,34],fit:28,flag:[21,26,27,28,29],flip:[33,34],folder:[31,33,34],follow:[28,31,33,34],foo:31,fork:31,format:31,framework:[33,34],free:31,from:[2,24,28,31],full:31,further:31,gener:[13,27,31,33,34],get:[2,33,34],get_classifier_for_attack:18,get_dataset:2,get_glove_emb:24,get_optim:13,get_output_filenam:21,get_root_dir:9,get_stopword:24,get_strategi:21,gigaword:24,git:[31,33,34],github:[31,33,34],githubusernam:31,given:31,global_step:13,glove:24,glove_fil:24,glove_semantic_similar:[0,12],glovesemanticsimilar:15,googl:[31,33,34],gpt2:[16,33,34],gpt2_gpu:[33,34],gpt2_gpu_id:16,gpt2_grammar_qu:[0,12],gpt2_pretrained_model:16,gpt2grammarqu:[16,33,34],gpu:[33,34],gpu_id:19,greatli:31,guid:[33,34],hack:31,has:[31,33,34],hash:2,have:[31,33,34],help:[31,33,34],helper:29,here:[31,33,34],highli:[33,34],histori:31,homepag:[33,34],how:[31,33,34],http:[24,33,34],hypothesi:[33,34],id2tok:24,id_to_tok:15,identical_strategi:[0,25],identicalstrategi:26,imdb:[33,34],immedi:[33,34],implement:28,includ:[28,31],incorrect:2,index:33,indic:[31,34],infer:[33,34],info:11,inform:[33,34],initi:[33,34],input:29,input_filenam:[3,5,7,8],input_fold:4,insid:31,instal:31,install_requir:31,instruct:[33,34],integ:[33,34],interf:[33,34],involv:31,issu:31,iter:31,iterabledataset:2,its:[24,31,33,34],json:[28,33,34],just:[26,31],karg:[13,14,15,16,17,18,19],keep:31,kind:31,lab:[33,34],label:[33,34],label_map:[33,34],languag:[33,34],last:31,latest:[33,34],learn:31,least:31,lei:30,leix:30,level:[0,11],librari:[31,33,34],like:31,line:[28,31],lint:31,list:[24,26,27,28,29,31,33,34],littl:31,load:24,load_detailed_result:24,load_glove_model:24,load_or_train_bert_clf:13,local:[31,33,34],locat:31,log:0,logger:11,look:31,lowercas:[33,34],lowest:2,machin:31,made:[33,34],mai:31,major:31,majority_aggregation_fn:18,make:[31,33,34],make_batch:16,make_input_output_pair:16,make_overview:[0,20],mani:31,map:[24,33,34],masked_lm:2,masked_lm_ratio:2,master:31,match:[33,34],md5:[2,9],md5_checksum:9,mean:[33,34],mean_aggregation_fn:18,meaning:[33,34],measur:[0,33,34],measure_qu:18,measurement_bas:[0,12,13,14,15,16,19],measurement_bundl:[18,21,26,27,28,29],measurement_util:[0,12],measurementbas:[13,14,15,16,17,19],measurementbundl:18,meet:31,memori:[33,34],merg:31,method:[29,31],metric:[33,34],might:31,minimum:[33,34],minor:31,mismatch:[33,34],miss:2,mit:[30,33,34],mkvirtualenv:31,mnli:[33,34],mnli_mi:[33,34],mock:31,model:[13,24,29,31,33,34],model_init:[2,13],model_nam:[18,33,34],model_wrapp:29,modelwrapp:29,modul:[31,33],more:[31,33,34],most:[26,27,28,29],multipl:[33,34],name:[2,11,24,31,33,34],narrow:31,natur:[33,34],nba:[33,34],necessari:[2,28,31],need:[26,27,28,29,33,34],neg:[33,34],neutral:[33,34],next:31,nlp:24,none:[9,30,31],normal:31,note:[31,33,34],now:31,num_paraphrases_per_text:[33,34],number:[26,27,28,29],numpi:24,object:[17,18,28],offici:31,old:31,omit:[33,34],onc:31,one:[26,27,28,29,31,33,34],onli:31,open:[31,33,34],oper:31,optim:13,option:[33,34],order:[31,33,34],origin:[26,31,33,34],other:[31,33,34],our:31,output:28,output_dir:[33,34],output_filenam:[3,4,5,7,8,18],outsid:31,overview:24,overview_result:24,overwrit:28,overwritten:[26,27,28,29],own:31,packag:31,page:[31,33],panda:24,param:13,paramet:[2,24,26,27,28,29],paraphras:[13,26,27,28,29,33,34],paraphrase_exampl:[26,27,28,29],paraphrase_field:[18,33,34],paraphrase_pred_accuracy_agg_fn:21,paraphrase_set:[21,28],paraphraseacc:[33,34],parser:28,part:31,parti:31,particular:[33,34],pass:31,patch:31,path:31,perform:31,phone:[33,34],pick:2,pip:[31,33,34],pipelin:[0,33,34],place:31,pleas:[28,31,33,34],point:31,posit:[33,34],possibl:[29,31,33,34],post:31,ppl_score:21,pre:31,predict:[13,33,34],predict_raw:13,prefix:[21,31],premis:[33,34],preprocess:[33,34],pretrain:24,process:[31,33,34],process_ag:[0,1],process_data:[3,4,5,7,8],process_imdb:[0,1],process_mnli:[0,1],process_mr:[0,1],process_raw:[33,34],process_snli:[0,1],process_yelp:[0,1],project:[24,31,33,34],proper:[33,34],properli:31,propos:31,provid:[33,34],publish:31,pull:[33,34],purpos:[33,34],push:31,put:31,pypi:31,pytest:31,python3:[33,34],python:[31,33,34],pytorch:[33,34],qualiti:[33,34],rais:2,random:[31,33,34],random_strategi:[0,25],randomli:27,randomstrategi:[27,33,34],raw:[33,34],read:[24,31],readi:31,recommend:[33,34],record:[26,27,28,29,33,34],refer:31,releas:[33,34],rememb:[31,33,34],repo:31,repositori:[33,34],reproduc:31,requir:31,research:[33,34],resolv:31,resource_util:0,result:[18,24,28],right:31,row:24,run:[28,31,33,34],run_evalu:13,sampl:2,save:[28,33,34],scenario:31,scheme:31,sci:[33,34],scope:31,script:[33,34],search:33,section:[31,33,34],see:[24,33,34],seed:2,self:28,semant:[33,34],send:31,sentenc:[14,26,27,33,34],sentiment:[33,34],separ:[31,33,34],seq:29,seri:[33,34],set:[2,28,31,33,34],set_context:29,setup:31,setup_custom_logg:11,sever:[31,33,34],sheet:[33,34],should:[26,27,28,29,31,33,34],show:[33,34],shuffl:27,similar:[31,33,34],sinc:31,site:[33,34],size:[2,24,33,34],snli:[33,34],softwar:[33,34],some:31,some_method:31,sometim:31,sourc:[2,3,4,5,6,7,8,9,11,13,14,15,16,17,18,19,21,23,24,26,27,28,29],specif:31,split:31,sport:[33,34],stabl:[31,33,34],stanford:24,start:[33,34],statu:31,std_aggregation_fn:18,step:[31,33,34],stopword:24,store:[26,27,28,29,33,34],strategi:[0,33,34],strategy_bas:[0,25,26,27,29],strategy_nam:21,strategybas:[26,27,28,29],strictli:[33,34],string:[24,26,27,28,29,33,34],style:31,subclass:[26,27,28,29],subdir:9,subsampl:[2,33,34],subsample_dataset:2,subsample_testset:[33,34],subset:31,suffix:21,suit:31,summari:13,support:31,sure:31,system:[31,33,34],tabl:[24,34],tag:31,take:[33,34],tech:[33,34],tensorflow:[33,34],test:[2,33,34],test_:31,test_a_modul:31,test_error:31,test_fibb:31,test_some_methed_input_non:31,test_some_method_1:31,test_some_method_timeout:31,test_some_method_value_error:31,testset:[13,21,33,34],text0:[13,33,34],text1:[13,33,34],text:[33,34],text_md5:2,textattack:29,textfooler_strategi:[0,25],textfoolerstrategi:29,than:[31,33,34],thei:31,them:[31,33,34],thi:[14,26,27,28,29,31,33,34],thing:31,third:31,through:[31,33,34],time:[33,34],titl:31,tmp:[33,34],tmp_output_filenam:28,tok2id:24,tok_to_id:15,tok_typ:29,token:[13,16,29],toks_list:16,tool:31,top:0,topic:[33,34],torch:2,tostr:29,train:[2,24,28,33,34],train_step:13,trainset:[13,21,28],troubleshoot:31,tutori:[33,34],two:[14,33,34],txt:[24,33,34],type:[24,33,34],uncas:[33,34],unittest:31,univers:[33,34],untar:9,unzip:9,updat:31,update_detailed_result:24,update_overview_result:24,upload:[31,33,34],url:9,usag:[33,34],use:[24,31,33,34],use_bert_clf_predict:18,use_editing_dist:18,use_glove_semantic_similar:18,use_gpt2_grammar_qu:18,use_gpu:[33,34],use_gpu_id:19,use_semantic_similar:[0,12],use_sim:21,use_use_semantic_simialr:18,used:[31,33,34],usernam:31,usesemanticsimilar:19,usesemanticsimilarit:[33,34],using:[31,33,34],util:2,valid:31,valu:[2,31,33,34],verifi:2,verify_dataset:2,version:[31,33,34],view:31,virtualenv:[31,33,34],virtualenvwrapp:31,volunt:31,wai:31,want:[31,33,34],warmup:13,web:31,websit:31,welcom:31,what:31,when:[2,31],whenev:31,where:31,whether:31,which:[31,33,34],whoever:31,why:30,wikipedia:24,within:2,won:[33,34],word:[24,27],work:[31,33,34],world:[33,34],would:31,wrapper:[2,29],write:24,yelp:[33,34],yet:30,you:[31,33,34],your:[31,33,34],your_name_her:31},titles:["fibber package","fibber.dataset package","fibber.dataset.dataset_utils module","fibber.dataset.process_ag module","fibber.dataset.process_imdb module","fibber.dataset.process_mnli module","fibber.dataset.process_mr module","fibber.dataset.process_snli module","fibber.dataset.process_yelp module","fibber.download_utils module","fibber.downloadable_resources module","fibber.log module","fibber.measurement package","fibber.measurement.bert_clf_prediction module","fibber.measurement.editing_distance module","fibber.measurement.glove_semantic_similarity module","fibber.measurement.gpt2_grammar_quality module","fibber.measurement.measurement_base module","fibber.measurement.measurement_utils module","fibber.measurement.use_semantic_similarity module","fibber.pipeline package","fibber.pipeline.benchmark module","fibber.pipeline.download_datasets module","fibber.pipeline.make_overview module","fibber.resource_utils module","fibber.strategy package","fibber.strategy.identical_strategy module","fibber.strategy.random_strategy module","fibber.strategy.strategy_base module","fibber.strategy.textfooler_strategy module","Credits","Contributing","History","Fibber","Fibber"],titleterms:{Use:[33,34],benchmark:[21,33,34],bert_clf_predict:13,bug:31,candid:31,content:[0,1,12,20,25],contribut:31,contributor:30,credit:30,dataset:[1,2,3,4,5,6,7,8,33,34],dataset_util:2,develop:30,document:31,download:[33,34],download_dataset:22,download_util:9,downloadable_resourc:10,editing_dist:14,featur:31,feedback:31,fibber:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,33,34],fix:31,format:[33,34],from:[33,34],get:31,glove_semantic_similar:15,gpt2_grammar_qu:16,guidelin:31,histori:32,identical_strategi:26,implement:31,indic:33,instal:[33,34],lead:30,log:11,make_overview:23,measur:[12,13,14,15,16,17,18,19],measurement_bas:17,measurement_util:18,modul:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],next:[33,34],overview:[33,34],packag:[0,1,12,20,25],pipelin:[20,21,22,23],process_ag:3,process_imdb:4,process_mnli:5,process_mr:6,process_snli:7,process_yelp:8,pull:31,pypi:[33,34],quickstart:[33,34],random_strategi:27,releas:31,report:31,request:31,requir:[33,34],resource_util:24,result:[33,34],sourc:[33,34],start:31,strategi:[25,26,27,28,29],strategy_bas:28,submit:31,submodul:[0,1,12,20,25],subpackag:0,tabl:33,test:31,textfooler_strategi:29,tip:31,type:31,unit:31,use_semantic_similar:19,what:[33,34],without:[33,34],workflow:31,write:31}})