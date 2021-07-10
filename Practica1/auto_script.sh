# Deshabilitad plugins
docker exec -u www-data -it servicio_nextcloud_75925767  php occ app:disable accessibility
docker exec -u www-data -it servicio_nextcloud_75925767  php occ app:disable dashboard
docker exec -u www-data -it servicio_nextcloud_75925767  php occ app:disable nextcloud_announcements
docker exec -u www-data -it servicio_nextcloud_75925767  php occ app:disable photos
docker exec -u www-data -it servicio_nextcloud_75925767  php occ app:disable weather_status
docker exec -u www-data -it servicio_nextcloud_75925767  php occ app:disable user_status
docker exec -u www-data -it servicio_nextcloud_75925767  php occ app:disable survey_client
docker exec -u www-data -it servicio_nextcloud_75925767  php occ app:disable support
docker exec -u www-data -it servicio_nextcloud_75925767  php occ app:disable recommendations
docker exec -u www-data -it servicio_nextcloud_75925767  php occ app:disable updatenotification

# Crear usuario con LDAP
docker exec servicio_openldap_75925767 ldapadd -x -D "cn=admin,dc=openstack,dc=org" -w password -c -f /files/open_config.ldif
# Cambiar contraseña a usuario con LDAP
docker exec servicio_openldap_75925767 ldappasswd -s carlosma7 -w password -D "cn=admin,dc=openstack,dc=org" -x "cn=carlosma7,ou=Users,dc=openstack,dc=org"
