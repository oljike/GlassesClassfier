from google_images_download import google_images_download   #importing the library


if __name__ == '__main__':

    response = google_images_download.googleimagesdownload()   #class instantiation
    arguments = {"keywords":"selfie with glasses female",
                 "limit":1000,
                 "print_urls":False,
                 "output_directory":'/home/oljike/PycharmProjects/GlassesClassification/google_pos',
                 'chromedriver':'/usr/lib/chromium-browser/chromedriver'}

    paths = response.download(arguments)   #passing the arguments to the function

