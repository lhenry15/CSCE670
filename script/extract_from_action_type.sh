# for extract action type from data
#cat ../data/train.csv | awk -F"," '{if(index($5, "clickout item")!=0){print $0}}' > ../preprocessed/clickItem
#cat ../data/train.csv | awk -F"," '{if(index($5, "search for item")!=0){print $0}}' > ../preprocessed/searchItem
#cat ../data/train.csv | awk -F"," '{if(index($5, "interaction item info")!=0){print $0}}' > ../preprocessed/interactionInfo
#cat ../data/train.csv | awk -F"," '{if(index($5, "interaction item rating")!=0){print $0}}' > ../preprocessed/interactionRating
#cat ../data/train.csv | awk -F"," '{if(index($5, "interaction item image")!=0){print $0}}' > ../preprocessed/interactionImage
cat ../data/train.csv | awk -F"," '{if(index($5, "interaction item deals")!=0){print $0}}' > ../preprocessed/interactionDeals

# for making the edgelist file
#cat sample.edge | cut -d" " -f1,2 | sort | uniq -c | awk '{print $2,$3,$1}' > deals.edgelist


# for filtering the column
#awk 'NR==FNR{a[$1];next}{ if($1 in a) print $0}' ../preprocessed/item.list mixed.rep > mixed_item.rep
