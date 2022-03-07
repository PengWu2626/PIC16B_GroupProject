import scrapy
from scrapy.spiders import Spider
from scrapy.http import Request

class ImdbSpider(scrapy.Spider):
    name = 'dog_spider'
    
# scrapy crawl dog_spider -o dogtime.csv 

# User-agent: *
# Disallow: /wp-content/plugins/pb-ads/ad-page/
# Disallow: /gnads/
# Disallow: /search.php
# Disallow: /*?hop

# Disallow: /pet-insurance

# Sitemap: http://dogtime.com/sitemap.xml

    # Dog Breeds
    start_urls = ['https://dogtime.com/dog-breeds/profiles/']

    deny_urls = [
        "/wp-content/plugins/pb-ads/ad-page/",
        "/gnads/",
        "/search.php",
        "/*?hop",
        "/pet-insurance"
    ]

    def parse(self, response):
        """
        This function starting at the website 'https://dogtime.com/dog-breeds/profiles/'
        , and get all dog breeds links.
        
        """
        for dog_link in response.css('div.list-item a::attr(href)').getall():
            yield Request(dog_link, callback = self.parse_dog)


    def parse_dog(self, response):
        """
        This function get each dog breed's chatacteristics.
        """
        valid_extensions = [".jpg"]
        ## get breed chatacteristics:

        name = response.css('h1::text').get()

        for characteristic in response.css('div.breed-characteristics-ratings-wrapper'):
            kind = characteristic.css('h3::text').get()
            characteristic_list = characteristic.css('div.characteristic-title ::text').getall()
            star_list =  characteristic.css('div.star ::text').getall()

            image_links = response.css('div.pbslideshow ::attr(data-lazy-src)').getall()
            image_link = [x for x in image_links if x.endswith(('jpg'))]
            print(image_link)

            if(not image_link):
                image_links = image_link = response.css('[id^=attachment] ::attr(data-lazy-src)').get()

            for i in range(len(characteristic_list)):
                yield{
                    "breed" : name,
                    "category" : kind,
                    "characteristic" : characteristic_list[i],
                    "star" : star_list[i],
                    "image_src" : image_link
                }