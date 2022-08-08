<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Add Posts</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-gH2yIJqKdNHPEq0n4Mqa/HGKIhSkIHeL5AyhkYV8i59U5AR6csBvApHHNl/vI1Bx" crossorigin="anonymous">
  </head>
  <body>
    <section style = "padding-top: 60px;">
        <div class = "container">
            <div class= "row">
              <div class = "col-md-6 offset-md-3">
                <div class = "card">
                  <div class= "card-header">
                    Add Post
                  </div>
                  <div class = "card-body">
                    <?php if(Session::has('post_created')): ?>
                    <div class="alert alert success" role="alert">
                      <?php echo e(Session::get('post_created')); ?>

                    </div>
                    <?php endif; ?>
                    <form method = "POST" action = "<?php echo e(route('post.create')); ?>">
                      <?php echo csrf_field(); ?> 
                      <div class = "form-group">
                        <label for = "title">Post Title</label>
                        <input type = "text" name = "title" class= "form-control" placeholder="Enter Post Title"/>
                      </div>

                      <div class="form-group">
                        <label for = "body">Post Description</label>
                        <textarea name = "body" class= "form-control" rows= "3"></textarea>
                      </div>
                      
                      <button type="submit" class="btn btn-success">Add Post</button> 
                    </form>
                  </div>
                </div>
              </div>
            </div>
        </div>    
    </section>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-A3rJD856KowSb7dwlZdYEkO39Gagi7vIsF0jrRAoQmDKKtQBHUuLZ9AsSv4jD4Xa" crossorigin="anonymous"></script>
  </body>
</html><?php /**PATH C:\Users\ADMIN\Desktop\php\example-app\resources\views/add-post.blade.php ENDPATH**/ ?>